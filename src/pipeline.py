"""End-to-end VF-50 pre-labeling pipeline (single image and batch directory)."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from .cvat_exporter import (
    ImagePrediction,
    export_cvat_xml,
    export_json,
    export_qc_csv,
)
from .detector import (
    BaseFaceDetector,
    FaceDetector,
    VALID_DETECTORS,
    rotate_bbox,
    rotate_image,
    unrotate_bbox,
    unrotate_points,
)
from .face_parser import FaceParserEngine
from .headpose_6drepnet import SixDRepNetEngine
from .landmark_engine import EnsembleVF50Engine, VF50Engine
from .scrfd_detector import SCRFDDetector
from .star_engine import STAREngine
from .tufa_engine import TUFAEngine
from .vf50_rules import PointAnnotation, QCFlag, VALID_VISIBILITY_MODES, apply_vf50_rules
from .vf50_schema import point_id_to_skeleton
from .visualizer import render_preview

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_ENSEMBLES = ("spiga_tufa_star", "spiga_tufa", "spiga_only")
VALID_HEADPOSES = ("6drepnet", "spiga")

_SCRFD_FILE = "scrfd_10g_bnkps.onnx"
_YUNET_FILE = "face_detection_yunet_2023mar.onnx"
_TUFA_WEIGHTS_FILE = "tufa_face_prompt_base.pth"
_TUFA_SHAPE_FILE = "tufa_shape_98.npz"
_STAR_WEIGHTS_FILE = "star_wflw.pkl"
_SIXD_FILE = "6DRepNet_300W_LP_AFLW2000.pth"
_FACE_PARSER_FILE = "bisenet_resnet34.onnx"


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Read image reliably on Windows even if path contains non-ASCII characters."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def collect_images(input_path: str) -> List[Path]:
    """Return sorted list of image Paths from a single file or directory."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


def list_available_weights(models_dir: str) -> List[str]:
    """List available SPIGA .pt checkpoints in models_dir, prioritizing IR fine-tuned weights."""
    if not os.path.isdir(models_dir):
        return []
    pt_files = sorted(
        f for f in os.listdir(models_dir)
        if f.lower().endswith(".pt") and os.path.isfile(os.path.join(models_dir, f))
    )
    priority = ["spiga_wflw_ir_vf50.pt", "spiga_wflw.pt"]
    ordered = [w for w in priority if w in pt_files]
    ordered.extend(w for w in pt_files if w not in priority)
    return ordered


def resolve_default_weights(models_dir: str) -> str:
    """Prefer fine-tuned IR weights when present, otherwise fall back to base spiga_wflw.pt."""
    if os.path.isfile(os.path.join(models_dir, "spiga_wflw_ir_vf50.pt")):
        return "spiga_wflw_ir_vf50.pt"
    return "spiga_wflw.pt"


def get_level2_capabilities(models_dir: str) -> Dict[str, Any]:
    """Inspect installed Level 2/3 model artifacts inside models_dir."""
    has_scrfd = os.path.isfile(os.path.join(models_dir, _SCRFD_FILE))
    has_yunet = os.path.isfile(os.path.join(models_dir, _YUNET_FILE))
    has_tufa = (
        os.path.isfile(os.path.join(models_dir, _TUFA_WEIGHTS_FILE))
        and os.path.isfile(os.path.join(models_dir, _TUFA_SHAPE_FILE))
    )
    has_star = os.path.isfile(os.path.join(models_dir, _STAR_WEIGHTS_FILE))
    has_6d = os.path.isfile(os.path.join(models_dir, _SIXD_FILE))
    has_parser = os.path.isfile(os.path.join(models_dir, _FACE_PARSER_FILE))

    detectors: List[str] = []
    if has_scrfd:
        detectors.append("scrfd10g")
    if has_yunet:
        detectors.append("yunet")

    ensembles: List[str] = []
    if has_tufa and has_star:
        ensembles.append("spiga_tufa_star")
    if has_tufa:
        ensembles.append("spiga_tufa")
    ensembles.append("spiga_only")

    default_ensemble = (
        "spiga_tufa_star"
        if (has_tufa and has_star)
        else ("spiga_tufa" if has_tufa else "spiga_only")
    )
    headposes = ["6drepnet", "spiga"] if has_6d else ["spiga"]

    return {
        "available_detectors": detectors or ["yunet"],
        "default_detector": "scrfd10g" if has_scrfd else "yunet",
        "available_ensembles": ensembles,
        "default_ensemble": default_ensemble,
        "available_headposes": headposes,
        "default_headpose": "6drepnet" if has_6d else "spiga",
        "level2_ready": bool(has_scrfd and has_tufa and has_6d),
        "star_ready": bool(has_star),
        "face_parser_ready": bool(has_parser),
        "available_visibility_modes": list(VALID_VISIBILITY_MODES),
        "default_visibility_mode": "auto",
    }


def get_level_presets(models_dir: str) -> Dict[str, Any]:
    """Return synchronized Level 1 / Level 2 / Level 3 stack presets for UI & CLI."""
    caps = get_level2_capabilities(models_dir)
    weights = set(list_available_weights(models_dir))
    has_base_spiga = "spiga_wflw.pt" in weights
    has_ir_spiga = "spiga_wflw_ir_vf50.pt" in weights
    has_yunet = "yunet" in caps["available_detectors"]
    has_l2 = bool(caps["level2_ready"])
    best_ens = "spiga_tufa_star" if (has_l2 and caps["star_ready"]) else "spiga_tufa"
    ens_label = "TUFA + STAR" if best_ens == "spiga_tufa_star" else "TUFA"
    fallback_w = resolve_default_weights(models_dir)

    presets = [
        {
            "id": "level1",
            "label": "Mức 1",
            "title": "Nhanh / Cơ bản",
            "description": "YuNet + SPIGA gốc (1-Pass, không TTA)",
            "available": bool(has_yunet and (has_base_spiga or has_ir_spiga)),
            "config": {
                "detector_mode": "yunet",
                "ensemble_mode": "spiga_only",
                "weights_name": "spiga_wflw.pt" if has_base_spiga else fallback_w,
                "headpose_mode": "spiga",
                "use_tta": False,
                "use_two_pass": False,
                "use_roll_align": True,
                "use_photo_preprocess": False,
            },
        },
        {
            "id": "level2",
            "label": "Mức 2",
            "title": "Mạnh nhất Pretrained",
            "description": f"SCRFD-10G + SPIGA gốc + {ens_label} + 6DRepNet + Flip TTA & 2-Pass",
            "available": bool(has_l2 and has_base_spiga),
            "config": {
                "detector_mode": "scrfd10g",
                "ensemble_mode": best_ens,
                "weights_name": "spiga_wflw.pt" if has_base_spiga else fallback_w,
                "headpose_mode": "6drepnet",
                "use_tta": True,
                "use_two_pass": True,
                "use_roll_align": True,
                "use_photo_preprocess": True,
            },
        },
        {
            "id": "level3",
            "label": "Mức 3 ★",
            "title": "Tối ưu Cabin IR (Mặc định)",
            "description": f"SCRFD-10G + SPIGA-IR Fine-tuned + {ens_label} + 6DRepNet + Flip TTA & 2-Pass",
            "available": bool(has_l2 and has_ir_spiga),
            "config": {
                "detector_mode": "scrfd10g",
                "ensemble_mode": best_ens,
                "weights_name": "spiga_wflw_ir_vf50.pt" if has_ir_spiga else fallback_w,
                "headpose_mode": "6drepnet",
                "use_tta": True,
                "use_two_pass": True,
                "use_roll_align": True,
                "use_photo_preprocess": True,
            },
        },
    ]
    default_level = "level3" if presets[2]["available"] else ("level2" if presets[1]["available"] else "level1")
    return {"level_presets": presets, "default_level": default_level}


class VF50Pipeline:
    """Coordinates Detectors (SCRFD-10G/YuNet), Ensemble (SPIGA+TUFA), 6DRepNet, and Exporters."""

    def __init__(
        self,
        models_dir: str,
        device: str = "cuda",
        det_score_threshold: float = 0.25,
        weights_name: Optional[str] = None,
        detector_mode: Optional[str] = None,
        ensemble_mode: Optional[str] = None,
        headpose_mode: Optional[str] = None,
        use_tta: bool = True,
        use_two_pass: bool = True,
        use_roll_align: bool = True,
        use_photo_preprocess: bool = True,
        visibility_mode: str = "auto",
    ) -> None:
        if visibility_mode not in VALID_VISIBILITY_MODES:
            raise ValueError(f"Unsupported visibility_mode: {visibility_mode!r}")
        self.models_dir = models_dir
        self.device = device
        self.det_score_threshold = det_score_threshold
        self.use_tta = use_tta
        self.use_two_pass = use_two_pass
        self.use_roll_align = use_roll_align
        self.use_photo_preprocess = use_photo_preprocess
        self.visibility_mode = visibility_mode

        caps = get_level2_capabilities(models_dir)
        self.default_detector_mode = detector_mode or caps["default_detector"]
        self.default_ensemble_mode = ensemble_mode or caps["default_ensemble"]
        self.default_headpose_mode = headpose_mode or caps["default_headpose"]

        yunet_path = os.path.join(models_dir, _YUNET_FILE)
        if not os.path.exists(yunet_path):
            raise FileNotFoundError(f"Missing YuNet model at: {yunet_path}")

        selected = weights_name or resolve_default_weights(models_dir)
        spiga_path = os.path.join(models_dir, selected)
        if not os.path.exists(spiga_path):
            raise FileNotFoundError(f"Missing SPIGA weights at: {spiga_path}")

        self.weights_name = selected
        self.detector = FaceDetector(yunet_path, score_threshold=det_score_threshold)
        self._scrfd_detector: Optional[SCRFDDetector] = None
        self._tufa_engine: Optional[TUFAEngine] = None
        self._star_engine: Optional[STAREngine] = None
        self._headpose_engine: Optional[SixDRepNetEngine] = None
        self._face_parser: Optional[FaceParserEngine] = None

        self.engine = VF50Engine(
            weights_dir=models_dir, device=device, weights_name=selected
        )

        # Eagerly initialize requested Level 2/3 components when available
        if self.default_detector_mode == "scrfd10g" and os.path.isfile(os.path.join(models_dir, _SCRFD_FILE)):
            self._get_detector("scrfd10g")
        if self.default_ensemble_mode in ("spiga_tufa", "spiga_tufa_star") and "spiga_tufa" in caps["available_ensembles"]:
            self._get_tufa()
        if self.default_ensemble_mode == "spiga_tufa_star" and "spiga_tufa_star" in caps["available_ensembles"]:
            self._get_star()
        if self.default_headpose_mode == "6drepnet" and "6drepnet" in caps["available_headposes"]:
            self._get_headpose()
        if self.visibility_mode == "auto" and caps["face_parser_ready"]:
            self._get_face_parser()

        self.ensemble = EnsembleVF50Engine(
            spiga_engine=self.engine,
            tufa_engine=self._tufa_engine,
            headpose_engine=self._headpose_engine,
            star_engine=self._star_engine,
            face_parser=self._face_parser,
        )

    def _get_detector(self, mode: str) -> BaseFaceDetector:
        if mode == "yunet":
            return self.detector
        if mode == "scrfd10g":
            scrfd_path = os.path.join(self.models_dir, _SCRFD_FILE)
            if os.path.isfile(scrfd_path):
                if self._scrfd_detector is None:
                    self._scrfd_detector = SCRFDDetector(
                        scrfd_path, fallback_yunet=self.detector
                    )
                return self._scrfd_detector
            return self.detector
        raise ValueError(f"Unsupported detector_mode: {mode} (valid: {VALID_DETECTORS})")

    def _get_tufa(self) -> Optional[TUFAEngine]:
        w_path = os.path.join(self.models_dir, _TUFA_WEIGHTS_FILE)
        s_path = os.path.join(self.models_dir, _TUFA_SHAPE_FILE)
        if os.path.isfile(w_path) and os.path.isfile(s_path):
            if self._tufa_engine is None:
                self._tufa_engine = TUFAEngine(w_path, s_path, device=self.device)
                if hasattr(self, "ensemble"):
                    self.ensemble.tufa_engine = self._tufa_engine
            return self._tufa_engine
        return None

    def _get_star(self) -> Optional[STAREngine]:
        s_path = os.path.join(self.models_dir, _STAR_WEIGHTS_FILE)
        if os.path.isfile(s_path):
            if self._star_engine is None:
                self._star_engine = STAREngine(s_path, device=self.device)
                if hasattr(self, "ensemble"):
                    self.ensemble.star_engine = self._star_engine
            return self._star_engine
        return None

    def _get_headpose(self) -> Optional[SixDRepNetEngine]:
        r_path = os.path.join(self.models_dir, _SIXD_FILE)
        if os.path.isfile(r_path):
            if self._headpose_engine is None:
                self._headpose_engine = SixDRepNetEngine(r_path, device=self.device)
                if hasattr(self, "ensemble"):
                    self.ensemble.headpose_engine = self._headpose_engine
            return self._headpose_engine
        return None

    def _get_face_parser(self) -> Optional[FaceParserEngine]:
        p_path = os.path.join(self.models_dir, _FACE_PARSER_FILE)
        if os.path.isfile(p_path):
            if self._face_parser is None:
                self._face_parser = FaceParserEngine(p_path)
                if hasattr(self, "ensemble"):
                    self.ensemble.face_parser = self._face_parser
            return self._face_parser
        return None

    def reload_spiga_weights(self, weights_name: str) -> None:
        """Hot-swap SPIGA weights without unloading TUFA, STAR, 6DRepNet, or SCRFD."""
        if weights_name == self.weights_name:
            return
        spiga_path = os.path.join(self.models_dir, weights_name)
        if not os.path.isfile(spiga_path):
            raise FileNotFoundError(f"Missing SPIGA weights at: {spiga_path}")
        self.engine.close()
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.engine = VF50Engine(
            weights_dir=self.models_dir, device=self.device, weights_name=weights_name
        )
        self.weights_name = weights_name
        self.ensemble.spiga_engine = self.engine

    def close(self) -> None:
        """Release all loaded GPU models."""
        if getattr(self, "engine", None) is not None:
            self.engine.close()
        if getattr(self, "_tufa_engine", None) is not None:
            self._tufa_engine.close()
            self._tufa_engine = None
        if getattr(self, "_star_engine", None) is not None:
            self._star_engine.close()
            self._star_engine = None
        if getattr(self, "_headpose_engine", None) is not None:
            self._headpose_engine.close()
            self._headpose_engine = None
        if getattr(self, "_face_parser", None) is not None:
            self._face_parser = None
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _no_face_prediction(
        name: str,
        orig_w: int,
        orig_h: int,
        detector_used: str = "scrfd10g",
        engine_used: str = "spiga_only",
        headpose_used: str = "spiga",
    ) -> ImagePrediction:
        empty_pts = [
            PointAnnotation(
                point_id=pid,
                skeleton=point_id_to_skeleton(pid),
                x=0.0,
                y=0.0,
                occluded=0,
                outside=1,
                confidence=0.0,
                uncertainty_px=0.0,
            )
            for pid in range(50)
        ]
        return ImagePrediction(
            image_name=name,
            width=orig_w,
            height=orig_h,
            rotation_applied="0",
            bbox_orig=None,
            face_score=0.0,
            used_clahe=True,
            multi_face=False,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            laplacian_var=0.0,
            points=empty_pts,
            qc_flags=[
                QCFlag(
                    "LOW_VISIBILITY",
                    "Không tự động detect được khuôn mặt; hãy vẽ khung mặt thủ công hoặc kiểm tra frame trống.",
                )
            ],
            detected=False,
            detector_used=detector_used,
            engine_used=engine_used,
            headpose_used=headpose_used,
            mean_confidence=0.0,
            max_uncertainty_px=0.0,
        )

    def predict_image(
        self,
        image_path: str,
        image_name: Optional[str] = None,
        rotate_mode: str = "0",
        driver_side: str = "largest",
        yaw_occluded: float = 35.0,
        yaw_outside: float = 55.0,
        manual_bbox: Optional[Sequence[float]] = None,
        detector_mode: Optional[str] = None,
        ensemble_mode: Optional[str] = None,
        headpose_mode: Optional[str] = None,
        use_tta: Optional[bool] = None,
        use_two_pass: Optional[bool] = None,
        use_roll_align: Optional[bool] = None,
        use_photo_preprocess: Optional[bool] = None,
        visibility_mode: Optional[str] = None,
    ) -> ImagePrediction:
        """Run full pre-labeling pipeline on a single image."""
        orig_img = imread_unicode(image_path)
        if orig_img is None:
            raise ValueError(f"Cannot decode image: {image_path}")

        orig_h, orig_w = orig_img.shape[:2]
        name = image_name or os.path.basename(image_path)

        det_mode = detector_mode or self.default_detector_mode
        ens_mode = ensemble_mode or self.default_ensemble_mode
        hp_mode = headpose_mode or self.default_headpose_mode
        vis_mode = visibility_mode or self.visibility_mode
        run_tta = self.use_tta if use_tta is None else bool(use_tta)
        run_two_pass = self.use_two_pass if use_two_pass is None else bool(use_two_pass)
        run_roll_align = self.use_roll_align if use_roll_align is None else bool(use_roll_align)
        run_photo_preprocess = (
            self.use_photo_preprocess if use_photo_preprocess is None else bool(use_photo_preprocess)
        )

        if ens_mode not in VALID_ENSEMBLES:
            raise ValueError(f"Unsupported ensemble_mode: {ens_mode}")
        if hp_mode not in VALID_HEADPOSES:
            raise ValueError(f"Unsupported headpose_mode: {hp_mode}")
        if vis_mode not in VALID_VISIBILITY_MODES:
            raise ValueError(f"Unsupported visibility_mode: {vis_mode}")

        detector = self._get_detector(det_mode)
        active_det_name = "scrfd10g" if isinstance(detector, SCRFDDetector) else "yunet"

        if ens_mode == "spiga_tufa_star":
            self._get_tufa()
            self._get_star()
        elif ens_mode == "spiga_tufa":
            self._get_tufa()
        if hp_mode == "6drepnet":
            self._get_headpose()
        if vis_mode == "auto":
            self._get_face_parser()

        det_landmarks5: Optional[List[List[float]]] = None
        if manual_bbox is not None:
            angle_used = "0" if rotate_mode == "auto" else rotate_mode
            rot_img = rotate_image(orig_img, angle_used)
            bbox_rot = rotate_bbox(manual_bbox, angle_used, orig_w, orig_h)
            face_score = 1.0
            used_clahe = False
            multi_face = False
            active_det_name = "manual"
        elif rotate_mode == "auto":
            candidates = detector.detect_candidates_all_rotations(
                orig_img, driver_side=driver_side
            )
            if not candidates:
                return self._no_face_prediction(
                    name, orig_w, orig_h, detector_used=active_det_name
                )

            def _is_in_bounds(det_bbox: Sequence[float], r_img: np.ndarray) -> bool:
                rh, rw = r_img.shape[:2]
                bx, by, bw, bh = det_bbox
                return bx >= 0 and by >= 0 and (bx + bw) <= rw and (by + bh) <= rh

            cand_zero = next(
                (c for c in candidates if c[0] == "0" and c[2].score >= 0.25 and _is_in_bounds(c[2].bbox, c[1])),
                None,
            )
            if cand_zero is not None:
                _, _, det0, _ = cand_zero
                strong_override = [
                    c for c in candidates
                    if c[0] != "0"
                    and _is_in_bounds(c[2].bbox, c[1])
                    and c[2].score >= 0.65
                    and c[2].score >= det0.score + 0.20
                ]
                if strong_override:
                    angle_used, rot_img, det, multi_face = max(strong_override, key=lambda c: c[2].score)
                else:
                    angle_used, rot_img, det, multi_face = cand_zero
                bbox_rot = det.bbox
                face_score = det.score
                used_clahe = det.used_clahe
                det_landmarks5 = det.landmarks5
            else:
                best_rank = None
                best_cand = None
                for ang, r_img, det, multi in candidates:
                    in_bounds_bonus = 0.10 if _is_in_bounds(det.bbox, r_img) else -0.10
                    cand_lm = self.engine.predict(r_img, det.bbox)
                    rank = det.score + in_bounds_bonus - 0.015 * abs(cand_lm.roll)
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_cand = (ang, r_img, det, multi)
                assert best_cand is not None
                angle_used, rot_img, det, multi_face = best_cand
                bbox_rot = det.bbox
                face_score = det.score
                used_clahe = det.used_clahe
                det_landmarks5 = det.landmarks5
        else:
            angle_used, rot_img, det, multi_face = detector.detect_with_rotation(
                orig_img, rotate_mode=rotate_mode, driver_side=driver_side
            )
            if det is None:
                return self._no_face_prediction(
                    name, orig_w, orig_h, detector_used=active_det_name
                )
            bbox_rot = det.bbox
            face_score = det.score
            used_clahe = det.used_clahe
            det_landmarks5 = det.landmarks5

        rot_h, rot_w = rot_img.shape[:2]
        lm_pred = self.ensemble.predict(
            rot_img,
            bbox_rot,
            use_tta=run_tta,
            use_two_pass=run_two_pass,
            use_ensemble=(ens_mode in ("spiga_tufa", "spiga_tufa_star")),
            use_6drepnet=(hp_mode == "6drepnet"),
            landmarks5=det_landmarks5,
            use_roll_align=run_roll_align,
            use_photo_preprocess=run_photo_preprocess,
            use_star=(ens_mode == "spiga_tufa_star"),
            run_face_parsing=(vis_mode == "auto"),
        )

        rule_out = apply_vf50_rules(
            points_50=lm_pred.points_50,
            img_w=rot_w,
            img_h=rot_h,
            yaw=lm_pred.yaw,
            face_score=face_score,
            laplacian_var=lm_pred.laplacian_var,
            face_w=lm_pred.face_width,
            face_h=lm_pred.face_height,
            used_clahe=used_clahe,
            multi_face=multi_face,
            yaw_occluded=yaw_occluded,
            yaw_outside=yaw_outside,
            point_confidences=lm_pred.confidence_50,
            point_uncertainties=lm_pred.uncertainty_px,
            pitch=lm_pred.pitch,
            visibility_mode=vis_mode,
            parsing_evidence=lm_pred.parsing_evidence,
        )

        # Unrotate 50 coordinates and bbox back to the original CVAT image frame
        rot_xy = np.array([[p.x, p.y] for p in rule_out.points], dtype=np.float64)
        orig_xy = unrotate_points(rot_xy, angle_used, orig_w, orig_h)
        for idx, pt in enumerate(rule_out.points):
            pt.x = float(orig_xy[idx, 0])
            pt.y = float(orig_xy[idx, 1])

        bbox_orig = unrotate_bbox(bbox_rot, angle_used, orig_w, orig_h)
        active_pts = [p for p in rule_out.points if p.outside == 0] or rule_out.points
        mean_conf = float(np.mean([p.confidence for p in active_pts]))
        max_unc = float(np.max([p.uncertainty_px for p in active_pts]))

        return ImagePrediction(
            image_name=name,
            width=orig_w,
            height=orig_h,
            rotation_applied=angle_used,
            bbox_orig=bbox_orig,
            face_score=face_score,
            used_clahe=used_clahe,
            multi_face=multi_face,
            yaw=rule_out.effective_yaw,
            pitch=lm_pred.pitch,
            roll=lm_pred.roll,
            laplacian_var=lm_pred.laplacian_var,
            points=rule_out.points,
            qc_flags=rule_out.qc_flags,
            detected=True,
            detector_used=active_det_name,
            engine_used=lm_pred.ensemble_used,
            headpose_used=lm_pred.headpose_source,
            mean_confidence=mean_conf,
            max_uncertainty_px=max_unc,
            preprocess_applied=lm_pred.preprocess_applied,
            visibility_mode_used=rule_out.visibility_mode_used,
            visibility_summary=rule_out.visibility_summary,
            face_3d=rule_out.face_3d_evidence.to_dict() if rule_out.face_3d_evidence is not None else None,
        )

    def run_batch(
        self,
        input_path: str,
        output_dir: str,
        rotate_mode: str = "0",
        driver_side: str = "largest",
        yaw_occluded: float = 35.0,
        yaw_outside: float = 55.0,
        template_xml_path: Optional[str] = None,
        save_previews: bool = False,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        detector_mode: Optional[str] = None,
        ensemble_mode: Optional[str] = None,
        headpose_mode: Optional[str] = None,
        use_tta: Optional[bool] = None,
        use_two_pass: Optional[bool] = None,
        use_roll_align: Optional[bool] = None,
        use_photo_preprocess: Optional[bool] = None,
        visibility_mode: Optional[str] = None,
    ) -> List[ImagePrediction]:
        """Run pre-labeling across a file or folder and write XML, JSON, CSV, and optional previews."""
        images = collect_images(input_path)
        if not images:
            raise ValueError(f"No images found in: {input_path}")

        os.makedirs(output_dir, exist_ok=True)
        previews_dir = os.path.join(output_dir, "previews")
        if save_previews:
            os.makedirs(previews_dir, exist_ok=True)

        predictions: List[ImagePrediction] = []
        total = len(images)
        for idx, img_path in enumerate(images, start=1):
            pred = self.predict_image(
                str(img_path),
                image_name=img_path.name,
                rotate_mode=rotate_mode,
                driver_side=driver_side,
                yaw_occluded=yaw_occluded,
                yaw_outside=yaw_outside,
                detector_mode=detector_mode,
                ensemble_mode=ensemble_mode,
                headpose_mode=headpose_mode,
                use_tta=use_tta,
                use_two_pass=use_two_pass,
                use_roll_align=use_roll_align,
                use_photo_preprocess=use_photo_preprocess,
                visibility_mode=visibility_mode,
            )
            predictions.append(pred)

            if save_previews:
                bgr = imread_unicode(str(img_path))
                if bgr is not None:
                    prev_img = render_preview(bgr, pred)
                    prev_out = os.path.join(previews_dir, f"{img_path.stem}_preview.png")
                    prev_img.save(prev_out)

            if progress_cb:
                progress_cb(idx, total, img_path.name)

        export_cvat_xml(
            predictions,
            os.path.join(output_dir, "annotations.xml"),
            template_xml_path=template_xml_path,
        )
        export_json(predictions, os.path.join(output_dir, "predictions.json"))
        export_qc_csv(predictions, os.path.join(output_dir, "qc_report.csv"))
        return predictions
