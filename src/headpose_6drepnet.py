"""6DRepNet (RepVGG-B1g2) headpose estimation engine with Flip TTA."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from vendor.SixDRepNet import get_RepVGG_func_by_name


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    v_mag = torch.sqrt(torch.sum(v * v, dim=1)).clamp_min(1e-8)
    return v / v_mag.unsqueeze(1)


def _compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    x = _normalize_vector(poses[:, 0:3])
    z = _normalize_vector(torch.cross(x, poses[:, 3:6], dim=1))
    y = torch.cross(z, x, dim=1)
    return torch.stack((x, y, z), dim=2)


def _compute_euler_angles_from_rotation_matrices(R: torch.Tensor) -> torch.Tensor:
    batch = R.shape[0]
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = (sy < 1e-6).to(R.dtype)

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = y
    zs = torch.zeros_like(z)

    out_euler = torch.zeros((batch, 3), dtype=R.dtype, device=R.device)
    out_euler[:, 0] = x * (1.0 - singular) + xs * singular
    out_euler[:, 1] = y * (1.0 - singular) + ys * singular
    out_euler[:, 2] = z * (1.0 - singular) + zs * singular
    return out_euler


class _SixDRepNet(nn.Module):
    """6DRepNet inference network matching thohemp/6DRepNet RepVGG-B1g2 weights."""

    def __init__(self, backbone_name: str = "RepVGG-B1g2", deploy: bool = True) -> None:
        super().__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy=deploy)
        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for name, mod in self.layer4.named_modules():
            if ("rbr_dense" in name or "rbr_reparam" in name) and isinstance(mod, nn.Conv2d):
                last_channel = mod.out_channels
        self.linear_reg = nn.Linear(last_channel, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return _compute_rotation_matrix_from_ortho6d(x)


class SixDRepNetEngine:
    """Runs 6DRepNet headpose inference aligned to VF-50/SPIGA yaw sign convention."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights_path: str, device: str = "cuda", yaw_sign: float = -1.0) -> None:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Missing 6DRepNet weights at: {weights_path}")
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.yaw_sign = float(yaw_sign)

        self.model = _SixDRepNet(backbone_name="RepVGG-B1g2", deploy=True)
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _crop_face(
        image_bgr: np.ndarray,
        bbox: Sequence[float],
        expand: float = 1.25,
        roll_deg: float = 0.0,
    ) -> np.ndarray:
        """Extract aspect-preserving padded square crop (de-rotated by eye roll) resized to 224x224 RGB."""
        bx, by, bw, bh = [float(v) for v in bbox]
        cx = bx + 0.5 * bw
        cy = by + 0.5 * bh
        side = max(bw, bh, 16.0) * expand
        scale = 224.0 / side
        if abs(roll_deg) >= 12.0:
            mat = cv2.getRotationMatrix2D((cx, cy), float(roll_deg), scale)
            mat[0, 2] += 112.0 - cx
            mat[1, 2] += 112.0 - cy
        else:
            mat = np.array(
                [
                    [scale, 0.0, 112.0 - scale * cx],
                    [0.0, scale, 112.0 - scale * cy],
                ],
                dtype=np.float32,
            )
        crop_bgr = cv2.warpAffine(
            image_bgr, mat, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, rgb_224: np.ndarray) -> np.ndarray:
        arr = rgb_224.astype(np.float32) / 255.0
        arr = (arr - self._MEAN[None, None, :]) / self._STD[None, None, :]
        return np.transpose(arr, (2, 0, 1))

    @torch.no_grad()
    def predict(
        self,
        image_bgr: np.ndarray,
        bbox: Sequence[float],
        use_flip_tta: bool = True,
        roll_deg: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Return (yaw, pitch, roll) in degrees, matching SPIGA convention (+yaw = face turned right)."""
        crop_rgb = self._crop_face(image_bgr, bbox, expand=1.25, roll_deg=roll_deg)
        if use_flip_tta:
            crop_flip = cv2.flip(crop_rgb, 1)
            batch_np = np.stack([self._to_tensor(crop_rgb), self._to_tensor(crop_flip)], axis=0)
        else:
            batch_np = self._to_tensor(crop_rgb)[None, ...]

        tensor = torch.from_numpy(batch_np).to(self.device)
        rot_mats = self.model(tensor)
        euler_deg = _compute_euler_angles_from_rotation_matrices(rot_mats) * (180.0 / np.pi)
        euler_np = euler_deg.float().cpu().numpy()

        # Model returns [pitch, yaw, roll]
        if use_flip_tta:
            pitch = 0.5 * float(euler_np[0, 0] + euler_np[1, 0])
            yaw = 0.5 * float(euler_np[0, 1] - euler_np[1, 1]) * self.yaw_sign
            roll = 0.5 * float(euler_np[0, 2] - euler_np[1, 2])
        else:
            pitch = float(euler_np[0, 0])
            yaw = float(euler_np[0, 1]) * self.yaw_sign
            roll = float(euler_np[0, 2])

        if abs(roll_deg) >= 12.0:
            roll = ((roll + float(roll_deg) + 180.0) % 360.0) - 180.0
        return yaw, pitch, roll

    def close(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        del model
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
