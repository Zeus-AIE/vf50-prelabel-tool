"""TUFA (Task-agnostic Unified Face Alignment, IJCV 2025) WFLW-98 inference engine."""

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

from vendor.TUFA.Prompt import Face_prompt

from .vf50_schema import flip_wflw98_points


def _get_dir(src_point: Sequence[float], rot_rad: float) -> Tuple[float, float]:
    sn, cs = float(np.sin(rot_rad)), float(np.cos(rot_rad))
    return (
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    )


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_tufa_affine(bbox: Sequence[float], scale_factor: float = 1.20, output_size: int = 256) -> np.ndarray:
    """Replicate TUFA/utils/get_transforms.py exact square affine transform."""
    bw = float(bbox[2])
    bh = float(bbox[3])
    side = max(bw, bh, 1.0)
    scale = (side / float(output_size)) * scale_factor

    src_w = scale * output_size
    dst_w = float(output_size)
    dst_h = float(output_size)

    src_dir = _get_dir([0.0, src_w * -0.5], 0.0)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    center = np.array([float(bbox[0]) + bw * 0.5, float(bbox[1]) + bh * 0.5], dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array(src_dir, dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir

    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    return cv2.getAffineTransform(src, dst)


def _inv_affine_points(coords: np.ndarray, trans: np.ndarray) -> np.ndarray:
    shifted = coords - trans[:, 2]
    inv_m = np.linalg.inv(trans[:, 0:2])
    return (inv_m @ shifted.T).T


class TUFAEngine:
    """Runs TUFA (DINO ViT-Small/8 + 6-layer Transformer Decoder) on WFLW-98."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        weights_path: str,
        shape_98_path: str,
        device: str = "cuda",
        scale_factor: float = 1.20,
    ) -> None:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Missing TUFA weights at: {weights_path}")
        if not os.path.isfile(shape_98_path):
            raise FileNotFoundError(f"Missing TUFA shape prompt at: {shape_98_path}")

        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.scale_factor = scale_factor
        self.amp_dtype = (
            torch.bfloat16
            if (use_cuda and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        self.model = Face_prompt("base")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.to(self.device)
        self.model.eval()

        offset = np.load(shape_98_path)["offset"] / 256.0
        self.prompt_98 = torch.from_numpy(offset).float().unsqueeze(0).to(self.device)

    def _to_tensor(self, rgb_256: np.ndarray) -> np.ndarray:
        arr = rgb_256.astype(np.float32) / 255.0
        arr = (arr - self._MEAN[None, None, :]) / self._STD[None, None, :]
        return np.transpose(arr, (2, 0, 1))

    @torch.no_grad()
    def predict_wflw98(
        self,
        image_bgr: np.ndarray,
        bbox: Sequence[float],
        use_flip_tta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (lms_98, flip_dist_98) in original upright image coordinates."""
        trans = _get_tufa_affine(bbox, scale_factor=self.scale_factor, output_size=256)
        crop_bgr = cv2.warpAffine(image_bgr, trans, (256, 256), flags=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if use_flip_tta:
            crop_flip = cv2.flip(crop_rgb, 1)
            batch_np = np.stack([self._to_tensor(crop_rgb), self._to_tensor(crop_flip)], axis=0)
            prompt = self.prompt_98.repeat(2, 1, 1)
        else:
            batch_np = self._to_tensor(crop_rgb)[None, ...]
            prompt = self.prompt_98

        tensor = torch.from_numpy(batch_np).to(self.device)
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda"), dtype=self.amp_dtype):
            outputs = self.model(tensor, prompt)

        preds_256 = outputs[:, -1, :, :].float().cpu().numpy() * 256.0
        pts0_orig = _inv_affine_points(preds_256[0], trans)
        if not use_flip_tta:
            return pts0_orig, np.zeros(98, dtype=np.float64)

        pts1_unflipped_256 = flip_wflw98_points(preds_256[1], 256.0, align_corners=False)
        pts1_orig = _inv_affine_points(pts1_unflipped_256, trans)
        lms_avg = 0.5 * (pts0_orig + pts1_orig)
        flip_dist_98 = np.linalg.norm(pts0_orig - pts1_orig, axis=1)
        return lms_avg, flip_dist_98

    def close(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        del model
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
