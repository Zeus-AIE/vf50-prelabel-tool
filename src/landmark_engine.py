"""SPIGA + TUFA + STAR + 6DRepNet landmark & headpose engine with 2-pass refinement, Flip TTA, and uncertainty."""

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

from .preprocessor import (
    conditional_photometric_enhance,
    estimate_roll_from_landmarks5,
    estimate_roll_from_wflw98,
    unwarp_points,
    warp_face_upright,
)
from .vf50_schema import VF50_TO_WFLW98, flip_wflw98_points

if TYPE_CHECKING:
    from .face_parser import FaceParserEngine, FaceParsingEvidence
    from .headpose_6drepnet import SixDRepNetEngine
    from .star_engine import STAREngine
    from .tufa_engine import TUFAEngine

DEFAULT_SPIGA_WEIGHT = 0.60
DEFAULT_ENSEMBLE3_WEIGHTS: Tuple[float, float, float] = (0.40, 0.30, 0.30)
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
    preprocess_applied: str = "none"
    parsing_evidence: Optional["FaceParsingEvidence"] = None


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
    """Landmark engine combining SPIGA-IR + TUFA + STAR + 6DRepNet + 2-Pass refinement + Flip TTA."""

    def __init__(
        self,
        spiga_engine: VF50Engine,
        tufa_engine: Optional[TUFAEngine] = None,
        headpose_engine: Optional[SixDRepNetEngine] = None,
        spiga_weight: float = DEFAULT_SPIGA_WEIGHT,
        star_engine: Optional[STAREngine] = None,
        ensemble3_weights: Tuple[float, float, float] = DEFAULT_ENSEMBLE3_WEIGHTS,
        face_parser: Optional[FaceParserEngine] = None,
    ) -> None:
        self.spiga_engine = spiga_engine
        self.tufa_engine = tufa_engine
        self.star_engine = star_engine
        self.headpose_engine = headpose_engine
        self.face_parser = face_parser
        self.spiga_weight = float(spiga_weight)
        self.ensemble3_weights = tuple(float(v) for v in ensemble3_weights)

    @staticmethod
    def _compute_spread_metrics(
        active_lms: Sequence[np.ndarray],
        active_flips: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (disagree_50, u_px) generalizing 2-model spread to K models continuously."""
        stack_50 = np.stack([m[_VF50_INDEX_ARRAY] for m in active_lms], axis=0)
        mean_50 = np.mean(stack_50, axis=0)
        rms_dev_50 = np.sqrt(np.mean(np.sum((stack_50 - mean_50[None, :, :]) ** 2, axis=2), axis=0))
        disagree_50 = 2.0 * rms_dev_50
        flip_spread_50 = np.mean([f[_VF50_INDEX_ARRAY] for f in active_flips], axis=0)
        u_px = np.sqrt(0.5 * (rms_dev_50 ** 2) + 0.5 * (0.5 * flip_spread_50) ** 2)
        return disagree_50, u_px

    @staticmethod
    def _weighted_sum_98(pairs: Sequence[Tuple[np.ndarray, float]]) -> np.ndarray:
        total_w = sum(w for _, w in pairs)
        out = np.zeros_like(pairs[0][0], dtype=np.float64)
        for arr, w in pairs:
            out += (w / total_w) * arr
        return out

    def _fuse_three_way(
        self,
        lms_spiga: np.ndarray,
        flip_spiga: np.ndarray,
        lms_tufa: np.ndarray,
        flip_tufa: np.ndarray,
        lms_star: np.ndarray,
        flip_star: np.ndarray,
        iod: float,
    ) -> Tuple[np.ndarray, str, np.ndarray, np.ndarray]:
        """3-way consensus & single-outlier rejection across SPIGA, TUFA, and STAR."""
        thr = 0.25 * iod
        d_st = float(np.median(np.linalg.norm(lms_spiga - lms_tufa, axis=1)))
        d_ss = float(np.median(np.linalg.norm(lms_spiga - lms_star, axis=1)))
        d_ts = float(np.median(np.linalg.norm(lms_tufa - lms_star, axis=1)))
        w_sp, w_tu, w_st = self.ensemble3_weights

        agree_st = d_st <= thr
        agree_ss = d_ss <= thr
        agree_ts = d_ts <= thr
        agree_cnt = int(agree_st) + int(agree_ss) + int(agree_ts)

        if agree_cnt >= 2:
            lms_98 = self._weighted_sum_98([(lms_spiga, w_sp), (lms_tufa, w_tu), (lms_star, w_st)])
            ensemble_used = "spiga+tufa+star"
            active_lms = (lms_spiga, lms_tufa, lms_star)
            active_flips = (flip_spiga, flip_tufa, flip_star)
        elif agree_ss:
            lms_98 = self._weighted_sum_98([(lms_spiga, w_sp), (lms_star, w_st)])
            ensemble_used = "spiga+star_fallback"
            active_lms = (lms_spiga, lms_star)
            active_flips = (flip_spiga, flip_star)
        elif agree_st:
            lms_98 = self._weighted_sum_98([(lms_spiga, w_sp), (lms_tufa, w_tu)])
            ensemble_used = "spiga+tufa_fallback"
            active_lms = (lms_spiga, lms_tufa)
            active_flips = (flip_spiga, flip_tufa)
        elif agree_ts:
            lms_98 = self._weighted_sum_98([(lms_tufa, w_tu), (lms_star, w_st)])
            ensemble_used = "tufa+star_fallback"
            active_lms = (lms_tufa, lms_star)
            active_flips = (flip_tufa, flip_star)
        else:
            lms_98 = lms_spiga
            ensemble_used = "spiga_fallback_outlier"
            active_lms = (lms_spiga, lms_tufa, lms_star)
            active_flips = (flip_spiga, flip_tufa, flip_star)

        disagree_50, u_px = self._compute_spread_metrics(active_lms, active_flips)
        return lms_98, ensemble_used, disagree_50, u_px

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
        landmarks5: Optional[Sequence[Sequence[float]]] = None,
        use_roll_align: bool = True,
        use_photo_preprocess: bool = True,
        use_star: bool = False,
        run_face_parsing: bool = False,
    ) -> LandmarkPrediction:
        img_h, img_w = image.shape[:2]
        orig_bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        tags: List[str] = []

        # 1. Conditional Photometric Guard (acts only on very dark or blown-out faces)
        work_img = image
        if use_photo_preprocess:
            work_img, photo_used, photo_tag = conditional_photometric_enhance(image, orig_bbox)
            if photo_used:
                tags.append(photo_tag)

        # 2. Roll-Aligned Face Warping (de-rotates in-plane roll before landmark & headpose models)
        align_roll_deg = 0.0
        lms_p1_raw: Optional[np.ndarray] = None
        if use_roll_align:
            if landmarks5 and len(landmarks5) >= 2:
                align_roll_deg = estimate_roll_from_landmarks5(landmarks5)
            else:
                lms_p1_raw, _, _, _, _ = self.spiga_engine.predict_wflw98_tta(
                    work_img, orig_bbox, use_flip_tta=False
                )
                align_roll_deg = estimate_roll_from_wflw98(lms_p1_raw)

        m_inv: Optional[np.ndarray] = None
        if use_roll_align and abs(align_roll_deg) >= 12.0:
            proc_img, proc_orig_bbox, m_inv = warp_face_upright(work_img, orig_bbox, align_roll_deg)
            proc_h, proc_w = proc_img.shape[:2]
            tags.append(f"roll({align_roll_deg:+.0f}°)")
        else:
            proc_img, proc_orig_bbox = work_img, orig_bbox
            proc_h, proc_w = img_h, img_w

        refined_bbox: Optional[List[float]] = None
        work_bbox = proc_orig_bbox
        if use_two_pass:
            if lms_p1_raw is not None and m_inv is None:
                lms_p1 = lms_p1_raw
            else:
                lms_p1, _, _, _, _ = self.spiga_engine.predict_wflw98_tta(
                    proc_img, proc_orig_bbox, use_flip_tta=False
                )
            cand_bbox = self._compute_tight_bbox(lms_p1, proc_orig_bbox, proc_w, proc_h)
            if cand_bbox is not None:
                refined_bbox = cand_bbox
                work_bbox = cand_bbox

        lms_spiga, flip_spiga, sp_yaw, sp_pitch, sp_roll = (
            self.spiga_engine.predict_wflw98_tta(proc_img, work_bbox, use_flip_tta=use_tta)
        )
        iod = max(
            float(np.linalg.norm(lms_spiga[60] - lms_spiga[72])),
            0.25 * work_bbox[2],
            16.0,
        )

        disagree_50: np.ndarray
        want_star = bool(use_ensemble and use_star and self.star_engine is not None)
        if want_star and self.tufa_engine is not None:
            lms_tufa, flip_tufa = self.tufa_engine.predict_wflw98(
                proc_img, work_bbox, use_flip_tta=use_tta
            )
            lms_star, flip_star = self.star_engine.predict_wflw98(
                proc_img, work_bbox, use_flip_tta=use_tta
            )
            lms_98, ensemble_used, disagree_50, u_px = self._fuse_three_way(
                lms_spiga, flip_spiga, lms_tufa, flip_tufa, lms_star, flip_star, iod
            )
        elif want_star:
            lms_star, flip_star = self.star_engine.predict_wflw98(
                proc_img, work_bbox, use_flip_tta=use_tta
            )
            disagree_98 = np.linalg.norm(lms_spiga - lms_star, axis=1)
            if float(np.median(disagree_98)) > 0.25 * iod:
                lms_98 = lms_spiga
                ensemble_used = "spiga_fallback_outlier"
            else:
                w_sp, _, w_st = self.ensemble3_weights
                lms_98 = self._weighted_sum_98([(lms_spiga, w_sp), (lms_star, w_st)])
                ensemble_used = "spiga+star"
            disagree_50, u_px = self._compute_spread_metrics(
                (lms_spiga, lms_star), (flip_spiga, flip_star)
            )
        elif use_ensemble and self.tufa_engine is not None:
            lms_tufa, flip_tufa = self.tufa_engine.predict_wflw98(
                proc_img, work_bbox, use_flip_tta=use_tta
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
            res_roll = estimate_roll_from_wflw98(lms_98)
            yaw, pitch, roll = self.headpose_engine.predict(
                proc_img, work_bbox, use_flip_tta=use_tta, roll_deg=res_roll
            )
            headpose_source = "6drepnet"
        else:
            yaw, pitch, roll = sp_yaw, sp_pitch, sp_roll
            headpose_source = "spiga"

        parsing_evidence: Optional[FaceParsingEvidence] = None
        if run_face_parsing and self.face_parser is not None:
            parsing_evidence = self.face_parser.analyze(proc_img, lms_98)

        if m_inv is not None:
            lms_98 = unwarp_points(lms_98, m_inv)
            roll = ((roll + align_roll_deg + 180.0) % 360.0) - 180.0

        lms_50 = lms_98[_VF50_INDEX_ARRAY, :].copy()
        sharpness = self.spiga_engine._compute_sharpness(image, orig_bbox)
        preprocess_applied = "+".join(tags) if tags else "none"

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
            refined_bbox=refined_bbox if m_inv is None else None,
            headpose_source=headpose_source,
            ensemble_used=ensemble_used,
            preprocess_applied=preprocess_applied,
            parsing_evidence=parsing_evidence,
        )
