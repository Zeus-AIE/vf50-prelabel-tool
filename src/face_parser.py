"""CelebAMask-HQ BiSeNet ResNet34 ONNX face parsing engine (CPU, 0 MB GPU VRAM)."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .vf50_schema import VF50_TO_WFLW98

# Exact 19-class CelebAMask-HQ IDs (verified against yakhyo/face-parsing prepare_labels.py)
BG = 0
SKIN = 1
L_BROW = 2
R_BROW = 3
L_EYE = 4
R_EYE = 5
EYE_G = 6
L_EAR = 7
R_EAR = 8
EAR_R = 9
NOSE = 10
MOUTH = 11
U_LIP = 12
L_LIP = 13
NECK = 14
NECK_L = 15
CLOTH = 16
HAIR = 17
HAT = 18

EYE_CLASSES = (L_EYE, R_EYE)
BROW_CLASSES = (L_BROW, R_BROW)
LIP_CLASSES = (MOUTH, U_LIP, L_LIP)
HAIR_HAT_CLASSES = (HAIR, HAT)
MASK_OBJ_CLASSES = (CLOTH, BG)

_VF50_INDEX_ARRAY = np.array(VF50_TO_WFLW98, dtype=np.int64)


@dataclass
class FaceParsingEvidence:
    valid: bool
    skin_ratio: float
    point_expected: np.ndarray      # (50,)
    point_skin: np.ndarray          # (50,)
    point_nose: np.ndarray          # (50,)
    point_eyeg: np.ndarray          # (50,)
    point_hair_hat: np.ndarray      # (50,)
    point_cloth_bg: np.ndarray      # (50,)
    point_cloth: np.ndarray         # (50,)
    point_bg: np.ndarray            # (50,)
    eye_l_vis_prob: float
    eye_r_vis_prob: float
    eye_l_eyeg_prob: float
    eye_r_eyeg_prob: float
    eye_l_nose_prob: float
    eye_r_nose_prob: float
    eye_l_hair_prob: float
    eye_r_hair_prob: float
    eye_l_bg_prob: float
    eye_r_bg_prob: float
    eye_l_contrast: float
    eye_r_contrast: float
    eye_l_glare_ratio: float
    eye_r_glare_ratio: float
    cheek_contrast: float
    lip_vis_prob: float
    lip_cloth_bg_prob: float
    lip_cloth_prob: float
    lip_bg_prob: float
    lip_skin_prob: float
    lip_contrast: float
    latency_ms: float


def _build_disk_offsets(radius: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    offs_y = []
    offs_x = []
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                offs_y.append(dy)
                offs_x.append(dx)
    return np.array(offs_y, dtype=np.int32), np.array(offs_x, dtype=np.int32)


_DISK_DY, _DISK_DX = _build_disk_offsets(5)
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


class FaceParserEngine:
    """Runs BiSeNet ResNet34 19-class face parsing on CPU via ONNXRuntime."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing BiSeNet face parser model at: {model_path}")
        self.model_path = model_path
        self.input_size = 512
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    @staticmethod
    def _build_parse_affine(lms_98: np.ndarray, size: int = 512) -> np.ndarray:
        eye_l = 0.5 * (lms_98[60] + lms_98[64])
        eye_r = 0.5 * (lms_98[68] + lms_98[72])
        v_eye = eye_r - eye_l
        roll_rad = math.atan2(float(v_eye[1]), float(v_eye[0])) if float(np.linalg.norm(v_eye)) >= 4.0 else 0.0

        center = np.mean(lms_98, axis=0)
        cx0, cy0 = float(center[0]), float(center[1])
        c, s = math.cos(roll_rad), math.sin(roll_rad)
        m_rot = np.array(
            [
                [c, s, cx0 - c * cx0 - s * cy0],
                [-s, c, cy0 + s * cx0 - c * cy0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rot_pts = (m_rot[:2, :2] @ lms_98.T).T + m_rot[:2, 2]
        min_xy = rot_pts.min(axis=0)
        max_xy = rot_pts.max(axis=0)
        fw = max(float(max_xy[0] - min_xy[0]), 16.0)
        fh = max(float(max_xy[1] - min_xy[1]), 16.0)

        cx_u = 0.5 * float(min_xy[0] + max_xy[0])
        cy_u = 0.5 * float(min_xy[1] + max_xy[1]) - 0.08 * fh
        side = max(fw, fh, 16.0) * 1.75
        scale = float(size) / side
        half = float(size) * 0.5

        m_crop = np.array(
            [
                [scale, 0.0, half - scale * cx_u],
                [0.0, scale, half - scale * cy_u],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return (m_crop @ m_rot)[:2, :]

    @staticmethod
    def _polygon_mask(pts: np.ndarray, size: int = 512) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.uint8)
        if len(pts) < 3 or not np.all(np.isfinite(pts)):
            return mask > 0
        poly = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly], 1)
        mask = cv2.dilate(mask, _DILATE_KERNEL, iterations=1)
        return mask > 0

    def analyze(self, image_bgr: np.ndarray, lms_98: np.ndarray) -> FaceParsingEvidence:
        """Align upright portrait crop to 512x512, run BiSeNet, and extract VF-50 occlusion evidence."""
        t0 = time.perf_counter()
        sz = self.input_size
        m_parse = self._build_parse_affine(lms_98, size=sz)
        crop_bgr = cv2.warpAffine(
            image_bgr,
            m_parse.astype(np.float32),
            (sz, sz),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose((crop_rgb - self._MEAN) / self._STD, (2, 0, 1))[None, ...].astype(np.float32)

        logits = self._session.run(None, {self._input_name: inp})[0][0]  # (19, 512, 512)
        logits_s = logits - np.max(logits, axis=0, keepdims=True)
        exp_l = np.exp(logits_s)
        probs = exp_l / np.sum(exp_l, axis=0, keepdims=True)

        p_skin = probs[SKIN]
        p_brow = probs[L_BROW] + probs[R_BROW]
        p_eye = probs[L_EYE] + probs[R_EYE]
        p_eyeg = probs[EYE_G]
        p_nose = probs[NOSE]
        p_lip = probs[MOUTH] + probs[U_LIP] + probs[L_LIP]
        p_hair_hat = probs[HAIR] + probs[HAT]
        p_cloth = probs[CLOTH]
        p_bg = probs[BG]
        p_cloth_bg = p_cloth + p_bg
        p_face_parts = p_skin + p_brow + p_eye + p_eyeg + p_nose + p_lip

        pts_98_p = (m_parse[:, :2] @ lms_98.T).T + m_parse[:, 2]
        pts_50_p = pts_98_p[_VF50_INDEX_ARRAY]

        point_expected = np.zeros(50, dtype=np.float64)
        point_skin = np.zeros(50, dtype=np.float64)
        point_nose = np.zeros(50, dtype=np.float64)
        point_eyeg = np.zeros(50, dtype=np.float64)
        point_hair_hat = np.zeros(50, dtype=np.float64)
        point_cloth_bg = np.zeros(50, dtype=np.float64)
        point_cloth = np.zeros(50, dtype=np.float64)
        point_bg = np.zeros(50, dtype=np.float64)

        for pid in range(50):
            ix = int(round(float(pts_50_p[pid, 0])))
            iy = int(round(float(pts_50_p[pid, 1])))
            xs = ix + _DISK_DX
            ys = iy + _DISK_DY
            inb = (xs >= 0) & (xs < sz) & (ys >= 0) & (ys < sz)
            if not np.any(inb):
                continue
            xs = xs[inb]
            ys = ys[inb]
            if pid <= 9:
                point_expected[pid] = float(np.mean(p_brow[ys, xs]))
            elif pid <= 13:
                point_expected[pid] = float(np.mean(p_nose[ys, xs]))
            elif pid <= 29:
                point_expected[pid] = float(np.mean(p_eye[ys, xs]))
            else:
                point_expected[pid] = float(np.mean(p_lip[ys, xs]))
            point_skin[pid] = float(np.mean(p_skin[ys, xs]))
            point_nose[pid] = float(np.mean(p_nose[ys, xs]))
            point_eyeg[pid] = float(np.mean(p_eyeg[ys, xs]))
            point_hair_hat[pid] = float(np.mean(p_hair_hat[ys, xs]))
            point_cloth_bg[pid] = float(np.mean(p_cloth_bg[ys, xs]))
            point_cloth[pid] = float(np.mean(p_cloth[ys, xs]))
            point_bg[pid] = float(np.mean(p_bg[ys, xs]))

        def _eye_stats(pts_poly: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
            m = self._polygon_mask(pts_poly, sz)
            if not np.any(m):
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            return (
                float(np.mean(p_eye[m])),
                float(np.mean(p_eyeg[m])),
                float(np.mean(p_nose[m])),
                float(np.mean(p_hair_hat[m])),
                float(np.mean(p_bg[m])),
                float(np.std(crop_gray[m])),
                float(np.mean(crop_gray[m] > 238.0)),
            )

        eye_l_vis, eye_l_eyeg, eye_l_nose, eye_l_hair, eye_l_bg, eye_l_ctr, eye_l_glare = _eye_stats(pts_50_p[14:22])
        eye_r_vis, eye_r_eyeg, eye_r_nose, eye_r_hair, eye_r_bg, eye_r_ctr, eye_r_glare = _eye_stats(pts_50_p[22:30])

        lip_mask = self._polygon_mask(pts_50_p[30:42], sz)
        if np.any(lip_mask):
            lip_vis = float(np.mean(p_lip[lip_mask]))
            lip_cloth_bg = float(np.mean(p_cloth_bg[lip_mask]))
            lip_cloth = float(np.mean(p_cloth[lip_mask]))
            lip_bg = float(np.mean(p_bg[lip_mask]))
            lip_skin = float(np.mean(p_skin[lip_mask]))
            lip_ctr = float(np.std(crop_gray[lip_mask]))
        else:
            lip_vis = lip_cloth_bg = lip_cloth = lip_bg = lip_skin = lip_ctr = 0.0

        # Cheek reference contrast sampled just below eyes
        cheek_vals = []
        y_top_chk = int(np.clip(max(float(pts_50_p[14:30, 1].max()), 0.0) + 10.0, 0, sz - 25))
        y_bot_chk = int(np.clip(min(float(pts_50_p[30:42, 1].min()) - 6.0, float(y_top_chk + 36)), y_top_chk + 8, sz))
        for eye_pts in (pts_50_p[14:22], pts_50_p[22:30]):
            ex_c = int(round(float(np.mean(eye_pts[:, 0]))))
            x1 = max(0, ex_c - 18)
            x2 = min(sz, ex_c + 18)
            if x2 > x1 + 4 and y_bot_chk > y_top_chk + 4:
                cheek_vals.append(crop_gray[y_top_chk:y_bot_chk, x1:x2].reshape(-1))
        cheek_contrast = float(np.std(np.concatenate(cheek_vals))) if cheek_vals else 0.0

        bx1 = int(np.clip(math.floor(float(pts_98_p[:, 0].min())), 0, sz - 2))
        by1 = int(np.clip(math.floor(float(pts_98_p[:, 1].min())), 0, sz - 2))
        bx2 = int(np.clip(math.ceil(float(pts_98_p[:, 0].max())), bx1 + 1, sz))
        by2 = int(np.clip(math.ceil(float(pts_98_p[:, 1].max())), by1 + 1, sz))
        skin_ratio = float(np.mean(p_skin[by1:by2, bx1:bx2]))
        face_area_parts = float(np.mean(p_face_parts[by1:by2, bx1:bx2]))

        max_eye_or_g = max(eye_l_vis + eye_l_eyeg, eye_r_vis + eye_r_eyeg)
        valid = (face_area_parts >= 0.25) and (max_eye_or_g >= 0.15 or lip_vis >= 0.25)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return FaceParsingEvidence(
            valid=valid,
            skin_ratio=skin_ratio,
            point_expected=point_expected,
            point_skin=point_skin,
            point_nose=point_nose,
            point_eyeg=point_eyeg,
            point_hair_hat=point_hair_hat,
            point_cloth_bg=point_cloth_bg,
            point_cloth=point_cloth,
            point_bg=point_bg,
            eye_l_vis_prob=eye_l_vis,
            eye_r_vis_prob=eye_r_vis,
            eye_l_eyeg_prob=eye_l_eyeg,
            eye_r_eyeg_prob=eye_r_eyeg,
            eye_l_nose_prob=eye_l_nose,
            eye_r_nose_prob=eye_r_nose,
            eye_l_hair_prob=eye_l_hair,
            eye_r_hair_prob=eye_r_hair,
            eye_l_bg_prob=eye_l_bg,
            eye_r_bg_prob=eye_r_bg,
            eye_l_contrast=eye_l_ctr,
            eye_r_contrast=eye_r_ctr,
            eye_l_glare_ratio=eye_l_glare,
            eye_r_glare_ratio=eye_r_glare,
            cheek_contrast=cheek_contrast,
            lip_vis_prob=lip_vis,
            lip_cloth_bg_prob=lip_cloth_bg,
            lip_cloth_prob=lip_cloth,
            lip_bg_prob=lip_bg,
            lip_skin_prob=lip_skin,
            lip_contrast=lip_ctr,
            latency_ms=latency_ms,
        )
