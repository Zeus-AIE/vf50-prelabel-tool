"""STAR (StackedHGNetV1 + CoordConv + AAM, CVPR 2023) WFLW-98 inference engine."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch

_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from vendor.STAR import STARWFLWConfig, StackedHGNetV1

from .vf50_schema import flip_wflw98_points


def _compute_side(bw: float, bh: float, side_mode: str) -> float:
    if side_mode == "max":
        return max(bw, bh, 1.0)
    if side_mode == "mean":
        return max(0.5 * (bw + bh), 1.0)
    if side_mode == "min":
        return max(min(bw, bh), 1.0)
    raise ValueError(f"Unsupported side_mode: {side_mode!r} (expected 'max', 'mean', or 'min')")


def _get_star_crop_matrix(
    bbox: Sequence[float],
    scale_factor: float = 1.0,
    side_mode: str = "max",
    image_size: int = 256,
) -> np.ndarray:
    """Build exact 3x3 perspective crop matrix matching STAR GetCropMatrix(align_corners=True)."""
    bx, by, bw, bh = [float(v) for v in bbox]
    center_w = bx + 0.5 * bw
    center_h = by + 0.5 * bh
    side = _compute_side(bw, bh, side_mode)
    scale = (side / 200.0) * float(scale_factor)
    scale_mu = float(image_size) / (scale * 200.0)
    to_c = (float(image_size) - 1.0) * 0.5  # 127.5 for 256x256
    return np.array(
        [
            [scale_mu, 0.0, to_c - scale_mu * center_w],
            [0.0, scale_mu, to_c - scale_mu * center_h],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _inv_perspective_points(coords: np.ndarray, inv_matrix: np.ndarray) -> np.ndarray:
    return (inv_matrix[:2, :2] @ coords.T).T + inv_matrix[:2, 2]


class STAREngine:
    """Runs STAR 4-Stack Hourglass + AAM on WFLW-98 in float32 precision."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        scale_factor: float = 1.0,
        side_mode: str = "max",
    ) -> None:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Missing STAR weights at: {weights_path}")

        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.scale_factor = float(scale_factor)
        self.side_mode = side_mode

        cfg = STARWFLWConfig()
        self.image_size = cfg.width
        self.model = StackedHGNetV1(
            config=cfg,
            classes_num=cfg.classes_num,
            edge_info=cfg.edge_info,
            nstack=cfg.nstack,
            nlevels=cfg.nlevels,
            in_channel=cfg.in_channel,
            increase=cfg.increase,
            add_coord=cfg.add_coord,
            decoder_type=cfg.decoder_type,
        )

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        cleaned = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.set_inference(True)
        self.model.float().to(self.device)
        self.model.eval()

    @staticmethod
    def _to_tensor_chw(crop_bgr: np.ndarray) -> np.ndarray:
        """Normalize BGR crop identically to STAR AlignmentDataset: (BGR / 255) * 2 - 1."""
        return crop_bgr.astype(np.float32).transpose(2, 0, 1) * (2.0 / 255.0) - 1.0

    @torch.no_grad()
    def predict_wflw98(
        self,
        image_bgr: np.ndarray,
        bbox: Sequence[float],
        use_flip_tta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (lms_98, flip_dist_98) in original upright image coordinates."""
        sz = self.image_size
        matrix = _get_star_crop_matrix(
            bbox,
            scale_factor=self.scale_factor,
            side_mode=self.side_mode,
            image_size=sz,
        )
        inv_matrix = np.linalg.inv(matrix)
        crop_bgr = cv2.warpPerspective(
            image_bgr,
            matrix.astype(np.float32),
            dsize=(sz, sz),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        if use_flip_tta:
            crop_flip = cv2.flip(crop_bgr, 1)
            batch_np = np.stack(
                [self._to_tensor_chw(crop_bgr), self._to_tensor_chw(crop_flip)],
                axis=0,
            )
        else:
            batch_np = self._to_tensor_chw(crop_bgr)[None, ...]

        tensor = torch.from_numpy(batch_np).to(self.device)
        _, _, landmarks = self.model(tensor)

        # Match STAR evaluate.py denorm_points(align_corners=False)
        preds_256 = ((landmarks.float().cpu().numpy() + 1.0) * float(sz) - 1.0) * 0.5
        pts0_orig = _inv_perspective_points(preds_256[0].astype(np.float64), inv_matrix)

        if not use_flip_tta:
            return pts0_orig, np.zeros(98, dtype=np.float64)

        pts1_unflipped_256 = flip_wflw98_points(preds_256[1], float(sz), align_corners=True)
        pts1_orig = _inv_perspective_points(pts1_unflipped_256, inv_matrix)
        lms_avg = 0.5 * (pts0_orig + pts1_orig)
        flip_dist_98 = np.linalg.norm(pts0_orig - pts1_orig, axis=1)
        return lms_avg, flip_dist_98

    def close(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        del model
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
