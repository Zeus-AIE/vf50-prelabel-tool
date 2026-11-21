"""VF-50 annotation rules engine: states (Visible/Occluded/Outside), topology, Table 3.1, and QC flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .vf50_schema import SKELETON_BY_NAME, SKELETON_SPECS, point_id_to_skeleton


@dataclass
class PointAnnotation:
    point_id: int
    skeleton: str
    x: float
    y: float
    occluded: int = 0  # 0 or 1
    outside: int = 0   # 0 or 1
    confidence: float = 1.0
    uncertainty_px: float = 0.0

    @property
    def state_name(self) -> str:
        if self.outside:
            return "outside"
        if self.occluded:
            return "occluded"
        return "visible"


@dataclass
class QCFlag:
    issue_type: str
    reason: str


@dataclass
class VF50RuleOutput:
    points: List[PointAnnotation]
    qc_flags: List[QCFlag] = field(default_factory=list)
    topology_adjusted: bool = False


def _interp_y_on_polyline(poly: np.ndarray, x_val: float) -> float:
    """Interpolate y at x_val along an x-sorted or nearly-monotonic polyline."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]
    return float(np.interp(x_val, xs_s, ys_s))


def enforce_topology(pts: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Enforce VF-50 geometric constraints in upright image coordinates."""
    out = pts.copy()
    adjusted = False

    # 1. Left & right eyelids: upper lid (15,16,17) must stay <= lower lid (21,20,19)
    for up_idx, lo_idx in ((15, 21), (16, 20), (17, 19), (23, 29), (24, 28), (25, 27)):
        if out[up_idx, 1] > out[lo_idx, 1]:
            mid_y = 0.5 * (out[up_idx, 1] + out[lo_idx, 1])
            out[up_idx, 1] = mid_y
            out[lo_idx, 1] = mid_y
            adjusted = True

    # 2. Outer mouth corners: left corner (30) must be left of right corner (36)
    if out[30, 0] > out[36, 0]:
        out[[30, 36]] = out[[36, 30]]
        adjusted = True

    x_left, x_right = float(out[30, 0]), float(out[36, 0])
    mouth_w = max(x_right - x_left, 4.0)

    # Prevent outer lip intermediate points from twisting past corners (extreme profile artifact)
    for idx in (31, 32, 33, 34, 35, 37, 38, 39, 40, 41):
        if out[idx, 0] < x_left:
            out[idx, 0] = x_left + 0.05 * mouth_w
            adjusted = True
        elif out[idx, 0] > x_right:
            # Allow slight natural lip protrusion only up to 2px; beyond that clamp to corner
            if out[idx, 0] > x_right + 2.0:
                out[idx, 0] = x_right
                adjusted = True

    # Ensure paired upper/lower outer lip columns don't cross vertically
    for up_idx, lo_idx in ((31, 41), (32, 40), (33, 39), (34, 38), (35, 37)):
        if out[up_idx, 1] > out[lo_idx, 1]:
            mid_y = 0.5 * (out[up_idx, 1] + out[lo_idx, 1])
            out[up_idx, 1] = mid_y
            out[lo_idx, 1] = mid_y
            adjusted = True

    # 3. Inner lip (42..49) strictly contained inside outer lip (30..41)
    margin_x = min(max(2.0, 0.05 * mouth_w), 0.2 * mouth_w)
    inner_left = x_left + margin_x
    inner_right = x_right - margin_x
    if inner_left >= inner_right:
        cx = 0.5 * (x_left + x_right)
        inner_left = cx - 1.0
        inner_right = cx + 1.0

    if out[42, 0] < inner_left or out[42, 0] > inner_right:
        out[42, 0] = np.clip(out[42, 0], inner_left, inner_right)
        adjusted = True
    if out[46, 0] > inner_right or out[46, 0] < out[42, 0]:
        out[46, 0] = np.clip(out[46, 0], max(out[42, 0] + 1.0, inner_left), inner_right)
        adjusted = True

    # Clamp horizontal span of inner upper/lower points within [42.x, 46.x]
    in_x_min, in_x_max = float(out[42, 0]), float(out[46, 0])
    for idx in (43, 44, 45, 47, 48, 49):
        if out[idx, 0] < in_x_min or out[idx, 0] > in_x_max:
            out[idx, 0] = np.clip(out[idx, 0], in_x_min, in_x_max)
            adjusted = True

    # Ensure paired inner lip upper/lower points don't cross vertically
    for up_idx, lo_idx in ((43, 49), (44, 48), (45, 47)):
        if out[up_idx, 1] > out[lo_idx, 1]:
            mid_y = 0.5 * (out[up_idx, 1] + out[lo_idx, 1])
            out[up_idx, 1] = mid_y
            out[lo_idx, 1] = mid_y
            adjusted = True

    # Vertically clamp all inner lip points inside upper/lower outer lip boundaries
    upper_outer = out[[30, 31, 32, 33, 34, 35, 36], :]
    lower_outer = out[[30, 41, 40, 39, 38, 37, 36], :]
    for idx in range(42, 50):
        px, py = float(out[idx, 0]), float(out[idx, 1])
        y_top = _interp_y_on_polyline(upper_outer, px)
        y_bot = _interp_y_on_polyline(lower_outer, px)
        pad = min(1.5, max(0.0, (y_bot - y_top) * 0.25))
        lo_bound = y_top + pad
        hi_bound = y_bot - pad
        if lo_bound <= hi_bound and (py < lo_bound or py > hi_bound):
            out[idx, 1] = np.clip(py, lo_bound, hi_bound)
            adjusted = True

    # Re-verify inner upper/lower order after vertical containment
    for up_idx, lo_idx in ((43, 49), (44, 48), (45, 47)):
        if out[up_idx, 1] > out[lo_idx, 1]:
            mid_y = 0.5 * (out[up_idx, 1] + out[lo_idx, 1])
            out[up_idx, 1] = mid_y
            out[lo_idx, 1] = mid_y
            adjusted = True

    return out, adjusted


def apply_vf50_rules(
    points_50: np.ndarray,
    img_w: int,
    img_h: int,
    yaw: float,
    face_score: float = 1.0,
    laplacian_var: float = 100.0,
    face_w: float = 200.0,
    face_h: float = 200.0,
    used_clahe: bool = False,
    multi_face: bool = False,
    yaw_occluded: float = 35.0,
    yaw_outside: float = 55.0,
    point_confidences: Optional[np.ndarray] = None,
    point_uncertainties: Optional[np.ndarray] = None,
) -> VF50RuleOutput:
    """Apply VF-50 topology, visibility states (Visible/Occluded/Outside), Table 3.1, and QC flags."""
    clean_pts, topology_adjusted = enforce_topology(points_50)

    occluded = np.zeros(50, dtype=int)
    outside = np.zeros(50, dtype=int)
    qc_flags: List[QCFlag] = []

    # 1. Out-of-bounds check
    for i in range(50):
        x, y = float(clean_pts[i, 0]), float(clean_pts[i, 1])
        if x < 0.0 or x >= float(img_w) or y < 0.0 or y >= float(img_h):
            outside[i] = 1

    # 2. Geometric foreshortening / yaw-based self-occlusion
    eye_l_w = max(float(abs(clean_pts[18, 0] - clean_pts[14, 0])), 1.0)
    eye_r_w = max(float(abs(clean_pts[26, 0] - clean_pts[22, 0])), 1.0)
    nose_x = float(clean_pts[10, 0])
    abs_yaw = abs(yaw)

    if yaw >= yaw_occluded:
        # Face turned toward viewer's right -> viewer's right side (matphai 22-29, longmayphai 5-9) is far side
        far_collapsed = (
            abs_yaw >= yaw_outside
            or (eye_r_w < 0.55 * eye_l_w)
            or (clean_pts[22, 0] < nose_x + 0.12 * eye_l_w)
        )
        far_ids = list(SKELETON_BY_NAME["longmayphai"].point_ids) + list(SKELETON_BY_NAME["matphai"].point_ids)
        if far_collapsed:
            for pid in far_ids:
                outside[pid] = 1
                occluded[pid] = 0
        else:
            for pid in far_ids:
                if outside[pid] == 0:
                    occluded[pid] = 1

        # Nasal bridge top (10) & far right mouth corner often partially occluded in strong right turn
        for pid in (10, 35, 36, 37, 46):
            if outside[pid] == 0:
                occluded[pid] = 1

    elif yaw <= -yaw_occluded:
        # Face turned toward viewer's left -> viewer's left side (mattrai 14-21, longmaytrai 0-4) is far side
        far_collapsed = (
            abs_yaw >= yaw_outside
            or (eye_l_w < 0.55 * eye_r_w)
            or (clean_pts[18, 0] > nose_x - 0.12 * eye_r_w)
        )
        far_ids = list(SKELETON_BY_NAME["longmaytrai"].point_ids) + list(SKELETON_BY_NAME["mattrai"].point_ids)
        if far_collapsed:
            for pid in far_ids:
                outside[pid] = 1
                occluded[pid] = 0
        else:
            for pid in far_ids:
                if outside[pid] == 0:
                    occluded[pid] = 1

        for pid in (10, 30, 31, 41, 42):
            if outside[pid] == 0:
                occluded[pid] = 1

    # 3. Table 3.1: if in-frame remaining points < minimum threshold for any skeleton, mark entire skeleton Outside=1
    for spec in SKELETON_SPECS:
        in_frame_count = sum(1 for pid in spec.point_ids if outside[pid] == 0)
        if 0 < in_frame_count < spec.min_visible_threshold:
            for pid in spec.point_ids:
                outside[pid] = 1
                occluded[pid] = 0

    # 4. Build PointAnnotation list
    annotations: List[PointAnnotation] = []
    for pid in range(50):
        conf = float(point_confidences[pid]) if point_confidences is not None else 1.0
        unc = float(point_uncertainties[pid]) if point_uncertainties is not None else 0.0
        annotations.append(
            PointAnnotation(
                point_id=pid,
                skeleton=point_id_to_skeleton(pid),
                x=float(clean_pts[pid, 0]),
                y=float(clean_pts[pid, 1]),
                occluded=int(occluded[pid]),
                outside=int(outside[pid]),
                confidence=round(conf, 4),
                uncertainty_px=round(unc, 2),
            )
        )

    # 5. QC Flags per Guideline Section 5
    low_conf_pts = [
        p for p in annotations
        if p.outside == 0 and p.occluded == 0 and p.confidence < 0.60
    ]
    if len(low_conf_pts) >= 2:
        low_conf_pts.sort(key=lambda p: p.confidence)
        top_str = ", ".join(
            f"ID {p.point_id} (±{p.uncertainty_px:.1f}px)" for p in low_conf_pts[:8]
        )
        qc_flags.append(
            QCFlag(
                "LOW_CONFIDENCE_POINTS",
                f"Độ đồng thuận model/TTA thấp tại {len(low_conf_pts)} điểm Visible ({top_str}); nên kiểm tra lại.",
            )
        )
    if multi_face:
        qc_flags.append(
            QCFlag("UNCERTAIN_SCOPE", "Phát hiện > 1 khuôn mặt trong ảnh; kiểm tra lại đúng người lái.")
        )

    low_vis_reasons: List[str] = []
    if laplacian_var < 25.0:
        low_vis_reasons.append(f"ảnh mờ (sharpness={laplacian_var:.1f})")
    if min(face_w, face_h) < 80.0:
        low_vis_reasons.append(f"mặt nhỏ ({face_w:.0f}x{face_h:.0f}px)")
    if face_score < 0.35:
        low_vis_reasons.append(f"score detect thấp ({face_score:.2f})")
    if used_clahe:
        low_vis_reasons.append("thiếu sáng (phải bật CLAHE mới detect được)")
    if low_vis_reasons:
        qc_flags.append(
            QCFlag("LOW_VISIBILITY", "Khó quan sát do: " + ", ".join(low_vis_reasons) + ".")
        )

    if abs_yaw >= yaw_occluded or topology_adjusted:
        reason_parts = []
        if abs_yaw >= yaw_occluded:
            reason_parts.append(f"góc nghiêng lớn (yaw={yaw:.1f}°)")
        if topology_adjusted:
            reason_parts.append("đã tự nắn topology mí mắt/môi")
        qc_flags.append(
            QCFlag("UNCERTAIN_BOUNDARY", "Cần kiểm tra ranh giới điểm: " + ", ".join(reason_parts) + ".")
        )

    if yaw_occluded <= abs_yaw < yaw_outside:
        qc_flags.append(
            QCFlag(
                "ATTRIBUTE_CHECK",
                f"Góc yaw={yaw:.1f}° nằm vùng giao thoa; kiểm tra lại nhãn Occluded hay Outside cho mắt/lông mày phía xa.",
            )
        )

    return VF50RuleOutput(points=annotations, qc_flags=qc_flags, topology_adjusted=topology_adjusted)
