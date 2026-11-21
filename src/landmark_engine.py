"""SPIGA + TUFA + 6DRepNet landmark & headpose engine with 2-pass refinement, Flip TTA, and uncertainty."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SPIGA_DIR = os.path.join(_ROOT, "vendor", "SPIGA")
if _SPIGA_DIR not in sys.path:
    sys.path.insert(0, _SPIGA_DIR)

import cv2
import numpy as np
import torch
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

from .vf50_schema import VF50_TO_WFLW98, flip_wflw98_points

if TYPE_CHECKING:
    from .headpose_6drepnet import SixDRepNetEngine
    from .tufa_engine import TUFAEngine

DEFAULT_SPIGA_WEIGHT = 0.60
_VF50_INDEX_ARRAY = np.array(VF50_TO_WFLW98, dtype=np.int64)


@dataclass
class LandmarkPrediction:
    points_50: np.ndarray          # (50, 2) float64 in upright image coordinates
    points_wflw98: np.ndarray      # (98, 2) float64
    yaw: float                     # degrees (+ = turned to viewer's right)
    pitch: float                   # degrees
    roll: float                    # degrees
    laplacian_var: float           # sharpness metric inside face bbox
    face_width: float
    face_height: float
    uncertainty_px: Optional[np.ndarray] = None
    confidence_50: Optional[np.ndarray] = None
    model_disagreement_px: Optional[np.ndarray] = None
    refined_bbox: Optional[List[float]] = None
    headpose_source: str = "spiga"
    ensemble_used: str = "spiga_only"


class VF50Engine:
    """Runs SPIGA WFLW-98 inference once loaded on GPU or CPU."""

    def __init__(
        self,
        weights_dir: str,
        device: str = "cuda",
        weights_name: str = "spiga_wflw.pt",
    ) -> None:
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        self.weights_name = weights_name

        cfg = ModelConfig("wflw", load_model_url=False)
        cfg.model_weights_path = weights_dir
        cfg.model_weights = weights_name
        self._framework = SPIGAFramework(cfg, gpus=[0] if use_cuda else [])

    def close(self) -> None:
        """Release model references and clear CUDA cache before switching weights."""
        framework = getattr(self, "_framework", None)
        self._framework = None
        del framework
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _compute_sharpness(image: np.ndarray, bbox: Sequence[float]) -> float:
        h, w = image.shape[:2]
        x, y, bw, bh = [int(round(v)) for v in bbox]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + max(bw, 1))
        y2 = min(h, y + max(bh, 1))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @torch.no_grad()
    def predict_wflw98_tta(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        use_flip_tta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Run SPIGA with optional horizontal Flip TTA -> (lms98, flip_dist98, yaw, pitch, roll)."""
        bbox_list = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        feat0 = self._framework.inference(image, [bbox_list])
        lms0 = np.array(feat0["landmarks"][0], dtype=np.float64)
        pose0 = feat0["headpose"][0]
        y0, p0, r0 = float(pose0[0]), float(pose0[1]), float(pose0[2])

        if not use_flip_tta:
            return lms0, np.zeros(98, dtype=np.float64), y0, p0, r0

        img_w = float(image.shape[1])
        bx, by, bw, bh = bbox_list
        img_flip = cv2.flip(image, 1)
        bbox_flip = [img_w - (bx + bw), by, bw, bh]
        feat1 = self._framework.inference(img_flip, [bbox_flip])
        lms1_flip = np.array(feat1["landmarks"][0], dtype=np.float64)
        lms1 = flip_wflw98_points(lms1_flip, img_w, align_corners=False)
        pose1 = feat1["headpose"][0]
        y1, p1, r1 = float(pose1[0]), float(pose1[1]), float(pose1[2])

        lms_avg = 0.5 * (lms0 + lms1)
        flip_dist = np.linalg.norm(lms0 - lms1, axis=1)
        return (
            lms_avg,
            flip_dist,
            0.5 * (y0 - y1),
            0.5 * (p0 + p1),
            0.5 * (r0 - r1),
        )

    def predict(self, image: np.ndarray, bbox: Sequence[float]) -> LandmarkPrediction:
        """Predict 50 VF-50 points and 3D headpose for a single face bbox [x, y, w, h]."""
        bbox_list = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        lms_98, _, yaw, pitch, roll = self.predict_wflw98_tta(
            image, bbox_list, use_flip_tta=False
        )
        lms_50 = lms_98[_VF50_INDEX_ARRAY, :].copy()
        sharpness = self._compute_sharpness(image, bbox_list)

        return LandmarkPrediction(
            points_50=lms_50,
            points_wflw98=lms_98,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            laplacian_var=sharpness,
            face_width=bbox_list[2],
            face_height=bbox_list[3],
        )


class EnsembleVF50Engine:
    """Level 2 engine combining SPIGA + TUFA + 6DRepNet + 2-Pass refinement + Flip TTA."""

    def __init__(
        self,
        spiga_engine: VF50Engine,
        tufa_engine: Optional[TUFAEngine] = None,
        headpose_engine: Optional[SixDRepNetEngine] = None,
        spiga_weight: float = DEFAULT_SPIGA_WEIGHT,
    ) -> None:
        self.spiga_engine = spiga_engine
        self.tufa_engine = tufa_engine
        self.headpose_engine = headpose_engine
        self.spiga_weight = float(spiga_weight)

    @staticmethod
    def _compute_tight_bbox(
        lms_98: np.ndarray, orig_bbox: Sequence[float], img_w: int, img_h: int
    ) -> Optional[List[float]]:
        _, _, ow, oh = [float(v) for v in orig_bbox]
        min_xy = lms_98.min(axis=0)
        max_xy = lms_98.max(axis=0)
        min_x, min_y = float(min_xy[0]), float(min_xy[1])
        max_x, max_y = float(max_xy[0]), float(max_xy[1])
        rw = max_x - min_x
        rh = max_y - min_y
        if not (0.45 * ow <= rw <= 1.8 * ow and 0.45 * oh <= rh <= 1.8 * oh):
            return None
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        if cx < 0.0 or cx > float(img_w) or cy < 0.0 or cy > float(img_h):
            return None
        return [min_x, min_y, rw, rh]

    def predict(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        use_tta: bool = True,
        use_two_pass: bool = True,
        use_ensemble: bool = True,
        use_6drepnet: bool = True,
    ) -> LandmarkPrediction:
        img_h, img_w = image.shape[:2]
        orig_bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

        refined_bbox: Optional[List[float]] = None
        work_bbox = orig_bbox
        if use_two_pass:
            lms_p1, _, _, _, _ = self.spiga_engine.predict_wflw98_tta(
                image, orig_bbox, use_flip_tta=False
            )
            cand_bbox = self._compute_tight_bbox(lms_p1, orig_bbox, img_w, img_h)
            if cand_bbox is not None:
                refined_bbox = cand_bbox
                work_bbox = cand_bbox

        lms_spiga, flip_spiga, sp_yaw, sp_pitch, sp_roll = (
            self.spiga_engine.predict_wflw98_tta(image, work_bbox, use_flip_tta=use_tta)
        )
        iod = max(
            float(np.linalg.norm(lms_spiga[60] - lms_spiga[72])),
            0.25 * work_bbox[2],
            16.0,
        )

        disagree_50: np.ndarray
        if use_ensemble and self.tufa_engine is not None:
            lms_tufa, flip_tufa = self.tufa_engine.predict_wflw98(
                image, work_bbox, use_flip_tta=use_tta
            )
            disagree_98 = np.linalg.norm(lms_spiga - lms_tufa, axis=1)
            disagree_50 = disagree_98[_VF50_INDEX_ARRAY]
            if float(np.median(disagree_98)) > 0.25 * iod:
                lms_98 = lms_spiga
                ensemble_used = "spiga_fallback_outlier"
            else:
                w = self.spiga_weight
                lms_98 = w * lms_spiga + (1.0 - w) * lms_tufa
                ensemble_used = "spiga+tufa"
            flip_spread_50 = 0.5 * (flip_spiga[_VF50_INDEX_ARRAY] + flip_tufa[_VF50_INDEX_ARRAY])
            u_px = np.sqrt(0.5 * (0.5 * disagree_50) ** 2 + 0.5 * (0.5 * flip_spread_50) ** 2)
        else:
            lms_98 = lms_spiga
            ensemble_used = "spiga_only"
            disagree_50 = np.zeros(50, dtype=np.float64)
            u_px = 0.5 * flip_spiga[_VF50_INDEX_ARRAY]

        conf_50 = np.clip(np.exp(-(u_px / iod) / 0.08), 0.0, 1.0)

        if use_6drepnet and self.headpose_engine is not None:
            eye_l_mid = 0.5 * (lms_98[60] + lms_98[64])
            eye_r_mid = 0.5 * (lms_98[68] + lms_98[72])
            v_eye = eye_r_mid - eye_l_mid
            eye_dist = float(np.linalg.norm(v_eye))
            face_roll_deg = float(np.degrees(np.arctan2(v_eye[1], v_eye[0]))) if eye_dist >= 8.0 else 0.0
            yaw, pitch, roll = self.headpose_engine.predict(
                image, work_bbox, use_flip_tta=use_tta, roll_deg=face_roll_deg
            )
            headpose_source = "6drepnet"
        else:
            yaw, pitch, roll = sp_yaw, sp_pitch, sp_roll
            headpose_source = "spiga"

        lms_50 = lms_98[_VF50_INDEX_ARRAY, :].copy()
        sharpness = self.spiga_engine._compute_sharpness(image, work_bbox)

        return LandmarkPrediction(
            points_50=lms_50,
            points_wflw98=lms_98,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            laplacian_var=sharpness,
            face_width=work_bbox[2],
            face_height=work_bbox[3],
            uncertainty_px=u_px,
            confidence_50=conf_50,
            model_disagreement_px=disagree_50,
            refined_bbox=refined_bbox,
            headpose_source=headpose_source,
            ensemble_used=ensemble_used,
        )
