"""SCRFD-10G ONNX face detector with CLAHE IR fallback and YuNet backup."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .detector import BaseFaceDetector, FaceDetection, FaceDetector, apply_clahe_bgr
from .preprocessor import apply_photometric_variant


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFDDetector(BaseFaceDetector):
    """InsightFace SCRFD-10G face detector (CPU ONNXRuntime, 0 MB VRAM)."""

    _FEAT_STRIDES = (8, 16, 32)
    _NUM_ANCHORS = 2

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.28,
        clahe_score_threshold: float = 0.22,
        nms_threshold: float = 0.4,
        min_face_size: int = 36,
        input_size: Tuple[int, int] = (640, 640),
        fallback_yunet: Optional[FaceDetector] = None,
    ) -> None:
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.clahe_score_threshold = clahe_score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self.input_size = input_size
        self.fallback_yunet = fallback_yunet

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        if len(outputs) != 9:
            raise ValueError(f"Expected 9 outputs in SCRFD ONNX model, got {len(outputs)}")
        self._anchor_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _get_anchors(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self._NUM_ANCHORS > 1:
            anchor_centers = np.stack([anchor_centers] * self._NUM_ANCHORS, axis=1).reshape((-1, 2))
        self._anchor_cache[key] = anchor_centers
        return anchor_centers

    def _run_scrfd(
        self, image: np.ndarray, threshold: float, used_clahe: bool
    ) -> List[FaceDetection]:
        h, w = image.shape[:2]
        input_w, input_h = self.input_size
        im_ratio = float(h) / max(float(w), 1.0)
        model_ratio = float(input_h) / float(input_w)
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = max(1, int(round(new_h / im_ratio)))
        else:
            new_w = input_w
            new_h = max(1, int(round(new_w * im_ratio)))

        det_scale = float(new_h) / max(float(h), 1.0)
        resized_img = cv2.resize(image, (new_w, new_h))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized_img

        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, (input_w, input_h), (127.5, 127.5, 127.5), swapRB=True
        )
        outs = self._session.run(None, {self._input_name: blob})

        fmc = len(self._FEAT_STRIDES)
        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []

        for idx, stride in enumerate(self._FEAT_STRIDES):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + fmc].reshape(-1, 4) * stride
            kps_preds = outs[idx + fmc * 2].reshape(-1, 10) * stride

            pos_inds = np.where(scores >= threshold)[0]
            if pos_inds.size == 0:
                continue

            fh = input_h // stride
            fw = input_w // stride
            anchors = self._get_anchors(fh, fw, stride)

            bboxes = _distance2bbox(anchors[pos_inds], bbox_preds[pos_inds]) / det_scale
            kpss = _distance2kps(anchors[pos_inds], kps_preds[pos_inds]).reshape(-1, 5, 2) / det_scale
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        if not scores_list:
            return []

        all_scores = np.concatenate(scores_list, axis=0)
        all_bboxes = np.concatenate(bboxes_list, axis=0)
        all_kpss = np.concatenate(kpss_list, axis=0)

        boxes_xywh = all_bboxes.copy()
        boxes_xywh[:, 2] = all_bboxes[:, 2] - all_bboxes[:, 0]
        boxes_xywh[:, 3] = all_bboxes[:, 3] - all_bboxes[:, 1]

        keep = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(),
            scores=all_scores.tolist(),
            score_threshold=0.0,
            nms_threshold=self.nms_threshold,
            eta=1.0,
            top_k=5000,
        )
        if keep is None or len(keep) == 0:
            return []

        keep_idxs = np.array(keep).flatten().tolist()
        keep_idxs.sort(key=lambda i: float(all_scores[i]), reverse=True)

        results: List[FaceDetection] = []
        for idx in keep_idxs:
            bx, by, bw, bh = [float(v) for v in boxes_xywh[idx]]
            if bw < self.min_face_size or bh < self.min_face_size:
                continue
            lm5 = [[float(kx), float(ky)] for kx, ky in all_kpss[idx]]
            results.append(
                FaceDetection(
                    bbox=[bx, by, bw, bh],
                    score=float(all_scores[idx]),
                    landmarks5=lm5,
                    used_clahe=used_clahe,
                )
            )
        return results

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Run SCRFD-10G on raw image first, rescuing low-score or dark/backlit frames via photometric variants."""
        raw_faces = self._run_scrfd(image, self.score_threshold, used_clahe=False)
        best_raw_score = max((f.score for f in raw_faces), default=0.0)
        if raw_faces and best_raw_score >= 0.48:
            return raw_faces

        best_faces = raw_faces
        best_score = best_raw_score

        def _consider(cand_faces: List[FaceDetection]) -> None:
            nonlocal best_faces, best_score
            if not cand_faces:
                return
            cand_score = max(f.score for f in cand_faces)
            if not best_faces or cand_score >= best_score + 0.06:
                best_faces = cand_faces
                best_score = cand_score

        clahe_img = apply_clahe_bgr(image)
        _consider(self._run_scrfd(clahe_img, self.clahe_score_threshold, used_clahe=True))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        p50_img = float(np.median(gray))
        primary_var = (
            "dark_lift"
            if p50_img < 90.0
            else ("highlight_recover" if p50_img > 165.0 else "shadow_highlight")
        )
        _consider(
            self._run_scrfd(
                apply_photometric_variant(image, primary_var),
                self.clahe_score_threshold,
                used_clahe=True,
            )
        )

        if not best_faces:
            secondary_var = "shadow_highlight" if primary_var != "shadow_highlight" else "dark_lift"
            _consider(
                self._run_scrfd(
                    apply_photometric_variant(image, secondary_var),
                    self.clahe_score_threshold,
                    used_clahe=True,
                )
            )

        if best_faces:
            return best_faces
        if self.fallback_yunet is not None:
            return self.fallback_yunet.detect_faces(image)
        return []
