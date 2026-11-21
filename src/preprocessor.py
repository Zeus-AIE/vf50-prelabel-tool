"""Adaptive preprocessing: Roll-Aligned face warping and Conditional Photometric Guard."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _normalize_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _gamma_lut(gamma: float) -> np.ndarray:
    inv = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip(np.power(inv, float(gamma)) * 255.0, 0.0, 255.0)
    return np.round(table).astype(np.uint8)


def _shadow_highlight_lut(
    shadow_gamma: float = 0.68,
    highlight_gamma: float = 1.28,
) -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    lift = np.power(x, float(shadow_gamma))
    comp = np.power(x, float(highlight_gamma))
    t = np.clip((x - 0.35) / 0.30, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    table = np.clip(((1.0 - w) * lift + w * comp) * 255.0, 0.0, 255.0)
    return np.round(table).astype(np.uint8)


def apply_photometric_variant(image_bgr: np.ndarray, variant: str) -> np.ndarray:
    """Apply luminance-only LAB photometric enhancement while preserving a/b color channels."""
    if variant in ("raw", "none"):
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    if variant == "dark_lift":
        l_g = cv2.LUT(l_ch, _gamma_lut(0.60))
        l_c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
        l_out = cv2.addWeighted(l_g, 0.60, l_c, 0.40, 0)
    elif variant == "dark_strong":
        l_g = cv2.LUT(l_ch, _gamma_lut(0.48))
        l_c = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(l_g)
        l_out = cv2.addWeighted(l_g, 0.55, l_c, 0.45, 0)
    elif variant == "shadow_highlight":
        l_b = cv2.LUT(l_ch, _shadow_highlight_lut(0.68, 1.28))
        l_c = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l_b)
        l_out = cv2.addWeighted(l_b, 0.60, l_c, 0.40, 0)
    elif variant == "highlight_recover":
        l_g = cv2.LUT(l_ch, _gamma_lut(1.40))
        l_c = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l_g)
        l_out = cv2.addWeighted(l_g, 0.70, l_c, 0.30, 0)
    elif variant == "low_contrast_stretch":
        lo, hi = np.percentile(l_ch, [2, 98])
        if hi > lo + 8.0:
            src_lo = max(float(lo) - 6.0, 0.0)
            src_hi = min(float(hi) + 6.0, 255.0)
            stretched = (l_ch.astype(np.float32) - src_lo) * (235.0 / max(src_hi - src_lo, 1.0)) + 10.0
            l_s = np.clip(np.round(stretched), 0.0, 255.0).astype(np.uint8)
        else:
            l_s = l_ch
        l_c = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l_s)
        l_out = cv2.addWeighted(l_s, 0.55, l_c, 0.45, 0)
    else:
        raise ValueError(f"Unsupported photometric variant: {variant!r}")

    return cv2.cvtColor(cv2.merge([l_out, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def estimate_roll_from_landmarks5(
    landmarks5: Optional[Sequence[Sequence[float]]],
) -> float:
    """Estimate in-plane roll angle (degrees) from 5-point face detector landmarks."""
    if not landmarks5 or len(landmarks5) < 2:
        return 0.0
    ex0, ey0 = float(landmarks5[0][0]), float(landmarks5[0][1])
    ex1, ey1 = float(landmarks5[1][0]), float(landmarks5[1][1])
    dx = ex1 - ex0
    dy = ey1 - ey0
    dist = math.hypot(dx, dy)
    if dist < 8.0:
        return 0.0

    angle = _normalize_deg(math.degrees(math.atan2(dy, dx)))
    # Verify eye-to-nose polarity to avoid 180-degree ambiguity on upside-down crops
    if len(landmarks5) >= 3:
        ux = dx / dist
        uy = dy / dist
        vx, vy = -uy, ux
        mx = 0.5 * (ex0 + ex1)
        my = 0.5 * (ey0 + ey1)
        nx, ny = float(landmarks5[2][0]), float(landmarks5[2][1])
        proj_down = (nx - mx) * vx + (ny - my) * vy
        if proj_down < -0.25 * dist:
            angle = _normalize_deg(angle + 180.0)
    return angle


def estimate_roll_from_wflw98(lms_98: np.ndarray) -> float:
    """Estimate in-plane eye roll angle (degrees) from WFLW-98 landmarks."""
    eye_l_mid = 0.5 * (lms_98[60] + lms_98[64])
    eye_r_mid = 0.5 * (lms_98[68] + lms_98[72])
    v_eye = eye_r_mid - eye_l_mid
    dist = float(np.linalg.norm(v_eye))
    if dist < 8.0:
        return 0.0
    return _normalize_deg(float(np.degrees(np.arctan2(float(v_eye[1]), float(v_eye[0])))))


def warp_face_upright(
    image: np.ndarray, bbox: Sequence[float], roll_deg: float
) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """Rotate local face ROI around bbox center so the eye line is horizontal."""
    bx, by, bw, bh = [float(v) for v in bbox]
    cx = bx + 0.5 * bw
    cy = by + 0.5 * bh
    out_size = int(max(256, round(max(bw, bh, 32.0) * 2.8)))
    half = 0.5 * float(out_size)

    m = cv2.getRotationMatrix2D((cx, cy), float(roll_deg), 1.0)
    m[0, 2] += half - cx
    m[1, 2] += half - cy

    warped = cv2.warpAffine(
        image, m, (out_size, out_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    warped_bbox = [half - 0.5 * bw, half - 0.5 * bh, bw, bh]
    m_inv = cv2.invertAffineTransform(m)
    return warped, warped_bbox, m_inv


def unwarp_points(pts: np.ndarray, m_inv: np.ndarray) -> np.ndarray:
    """Transform (N, 2) points from upright warped ROI back into original image coordinates."""
    arr = np.asarray(pts, dtype=np.float64)
    ones = np.ones((arr.shape[0], 1), dtype=np.float64)
    return np.hstack([arr, ones]) @ np.asarray(m_inv, dtype=np.float64).T


def analyze_face_photometrics(
    image_bgr: np.ndarray, bbox: Sequence[float]
) -> Dict[str, float]:
    """Compute LAB luminance distribution & directional asymmetry inside expanded face ROI."""
    h, w = image_bgr.shape[:2]
    bx, by, bw, bh = [float(v) for v in bbox]
    cx = bx + 0.5 * bw
    cy = by + 0.5 * bh
    half_w = 0.5 * max(bw, 1.0) * 1.15
    half_h = 0.5 * max(bh, 1.0) * 1.15

    x1 = max(0, int(math.floor(cx - half_w)))
    y1 = max(0, int(math.floor(cy - half_h)))
    x2 = min(w, int(math.ceil(cx + half_w)))
    y2 = min(h, int(math.ceil(cy + half_h)))
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return {
            "valid": 0.0,
            "p5": 0.0,
            "p50": 128.0,
            "p95": 255.0,
            "dark_frac": 0.0,
            "bright_frac": 0.0,
            "dyn_range": 128.0,
            "lr_asym": 0.0,
            "tb_asym": 0.0,
        }

    roi_bgr = image_bgr[y1:y2, x1:x2]
    roi_l = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    p5, p50, p95 = [float(v) for v in np.percentile(roi_l, [5, 50, 95])]
    mid_x = roi_l.shape[1] // 2
    mid_y = roi_l.shape[0] // 2
    lr_asym = abs(float(np.mean(roi_l[:, :mid_x])) - float(np.mean(roi_l[:, mid_x:])))
    tb_asym = abs(float(np.mean(roi_l[:mid_y, :])) - float(np.mean(roi_l[mid_y:, :])))
    return {
        "valid": 1.0,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "dark_frac": float(np.mean(roi_l <= 12)),
        "bright_frac": float(np.mean(roi_l >= 248)),
        "dyn_range": float(p95 - p5),
        "lr_asym": lr_asym,
        "tb_asym": tb_asym,
    }


def select_tta_exposure_variants(metrics: Dict[str, float], base_tag: str) -> List[str]:
    """Select 2 complementary photometric exposure variants that do not duplicate base_tag."""
    if base_tag == "dark_ir_lift":
        return ["dark_strong", "shadow_highlight"]
    if base_tag == "highlight_compress":
        return ["shadow_highlight", "highlight_recover"]
    if base_tag == "shadow_highlight":
        return ["dark_lift", "highlight_recover"]
    if metrics["p50"] < 95.0 or metrics["dark_frac"] >= 0.15:
        return ["shadow_highlight", "dark_lift"]
    if metrics["p50"] > 140.0 or metrics["bright_frac"] >= 0.06 or metrics["p95"] >= 225.0:
        return ["highlight_recover", "shadow_highlight"]
    return ["shadow_highlight", "dark_lift"]


def conditional_photometric_enhance(
    image_bgr: np.ndarray, bbox: Sequence[float]
) -> Tuple[np.ndarray, bool, str]:
    """Adaptively enhance dark, blown-out, split-lit backlit, or flat contrast faces."""
    m = analyze_face_photometrics(image_bgr, bbox)
    if m["valid"] < 0.5:
        return image_bgr, False, "none"

    p5 = m["p5"]
    p50 = m["p50"]
    p95 = m["p95"]
    dark_frac = m["dark_frac"]
    bright_frac = m["bright_frac"]
    dyn_range = m["dyn_range"]
    max_asym = max(m["lr_asym"], m["tb_asym"])

    is_split_lit = (
        (dark_frac >= 0.12 and bright_frac >= 0.06)
        or (p50 >= 50.0 and max_asym >= 42.0 and p5 <= 25.0 and p95 >= 215.0)
    )
    is_dark = (not is_split_lit) and (
        (p50 < 48.0) or (dark_frac > 0.35) or (dyn_range < 55.0 and p50 < 140.0)
    )
    is_bright = (not is_split_lit) and (not is_dark) and (p50 > 185.0 or bright_frac > 0.18)
    is_low_contrast = (not is_split_lit) and (not is_dark) and (not is_bright) and (dyn_range < 48.0)

    if is_split_lit:
        return apply_photometric_variant(image_bgr, "shadow_highlight"), True, "shadow_highlight"

    if is_dark:
        p50_safe = float(np.clip(p50, 12.0, 140.0))
        gamma = float(np.clip(math.log(95.0 / 255.0) / math.log(p50_safe / 255.0), 0.55, 0.85))
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_gamma = cv2.LUT(l_ch, _gamma_lut(gamma))
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_ch)
        l_out = cv2.addWeighted(l_gamma, 0.65, l_clahe, 0.35, 0)
        out = cv2.cvtColor(cv2.merge([l_out, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        return out, True, "dark_ir_lift"

    if is_bright:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_out = cv2.LUT(l_ch, _gamma_lut(1.35))
        out = cv2.cvtColor(cv2.merge([l_out, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        return out, True, "highlight_compress"

    if is_low_contrast:
        return apply_photometric_variant(image_bgr, "low_contrast_stretch"), True, "low_contrast_stretch"

    return image_bgr, False, "none"
