"""VF-50 annotation rules engine: states (Visible/Occluded/Outside), topology, Table 3.1, and QC flags."""

from __future__ import annotations

import math
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
    effective_yaw: float = 0.0
    face_roll_deg: float = 0.0


def _interp_y_on_polyline(poly: np.ndarray, x_val: float) -> float:
    """Interpolate y at x_val along an x-sorted or nearly-monotonic polyline."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]
    return float(np.interp(x_val, xs_s, ys_s))


def _compute_face_frame(pts: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute rotation-invariant local eye-aligned face coordinate axes (center, roll_rad, ux, uy)."""
    eye_l_mid = 0.5 * (pts[14] + pts[18])
    eye_r_mid = 0.5 * (pts[22] + pts[26])
    v_eye = eye_r_mid - eye_l_mid
    dist = float(np.linalg.norm(v_eye))
    center = np.mean(pts, axis=0)
    if dist < 4.0:
        return center, 0.0, np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)
    roll_rad = float(math.atan2(float(v_eye[1]), float(v_eye[0])))
    ux = v_eye / dist
    uy = np.array([-ux[1], ux[0]], dtype=np.float64)
    return center, roll_rad, ux, uy


def _rotate_points_around(pts: np.ndarray, center: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate Nx2 points around center by angle_rad in image coordinates."""
    if abs(angle_rad) < 1e-4:
        return pts.copy()
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot_mat = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts - center) @ rot_mat.T + center


def _to_face_upright(pts: np.ndarray, center: np.ndarray, roll_rad: float) -> np.ndarray:
    """De-rotate points so the eye line becomes horizontal (+X left->right)."""
    return _rotate_points_around(pts, center, -roll_rad)


def _from_face_upright(pts_local: np.ndarray, center: np.ndarray, roll_rad: float) -> np.ndarray:
    """Map upright local face points back into original image coordinates."""
    return _rotate_points_around(pts_local, center, roll_rad)


def enforce_topology(pts: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Enforce VF-50 geometric constraints in local eye-aligned upright face coordinates."""
    center, roll_rad, _, _ = _compute_face_frame(pts)
    out = _to_face_upright(pts, center, roll_rad)
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
        elif out[idx, 0] > x_right + 2.0:
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

    if not adjusted:
        return pts.copy(), False
    return _from_face_upright(out, center, roll_rad), True


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
    """Apply VF-50 topology, roll-invariant headpose check, visibility states, Table 3.1, and QC flags."""
    clean_pts, topology_adjusted = enforce_topology(points_50)
    center, roll_rad, _, _ = _compute_face_frame(clean_pts)
    face_roll_deg = float(math.degrees(roll_rad))
    local_pts = _to_face_upright(clean_pts, center, roll_rad)

    # Geometric asymmetry in eye-aligned upright face coordinates (roll-invariant)
    eye_l_w = max(float(np.linalg.norm(clean_pts[18] - clean_pts[14])), 1.0)
    eye_r_w = max(float(np.linalg.norm(clean_pts[26] - clean_pts[22])), 1.0)
    eye_asym = (eye_l_w - eye_r_w) / max(eye_l_w, eye_r_w, 1.0)  # >0 => face turned right

    nose_top_x = float(local_pts[10, 0])
    eye_l_mid_x = 0.5 * float(local_pts[14, 0] + local_pts[18, 0])
    eye_r_mid_x = 0.5 * float(local_pts[22, 0] + local_pts[26, 0])
    d_l = max(nose_top_x - eye_l_mid_x, 0.0)
    d_r = max(eye_r_mid_x - nose_top_x, 0.0)
    d_asym = (d_l - d_r) / max(d_l + d_r, 1.0)

    inter_outer = max(float(local_pts[26, 0] - local_pts[14, 0]), 1.0)
    eye_mid_x = 0.5 * float(local_pts[14, 0] + local_pts[26, 0])
    nose_shift = float(np.clip((local_pts[13, 0] - eye_mid_x) / inter_outer, -1.0, 1.0))

    geo_score = 0.45 * d_asym + 0.35 * eye_asym + 0.20 * nose_shift
    geo_yaw = float(np.clip(geo_score * 105.0, -80.0, 80.0))

    effective_yaw = float(yaw)
    yaw_geo_conflict = False
    if abs(geo_score) >= 0.22 and (eye_asym * d_asym > 0.0):
        if effective_yaw * geo_score < 0.0:
            yaw_geo_conflict = True
            effective_yaw = geo_yaw
        elif abs(geo_score) >= 0.40 and min(eye_l_w, eye_r_w) < 0.58 * max(eye_l_w, eye_r_w):
            effective_yaw = math.copysign(max(abs(effective_yaw), abs(geo_yaw)), geo_score)

    occluded = np.zeros(50, dtype=int)
    outside = np.zeros(50, dtype=int)
    qc_flags: List[QCFlag] = []

    # 1. Out-of-bounds check (outside CVAT canvas MUST be Outside=1, Occluded=0)
    for i in range(50):
        x, y = float(clean_pts[i, 0]), float(clean_pts[i, 1])
        if x < 0.0 or x >= float(img_w) or y < 0.0 or y >= float(img_h):
            outside[i] = 1

    # 2. Geometric foreshortening / yaw-based self-occlusion in upright face frame
    abs_yaw = abs(effective_yaw)
    if effective_yaw >= yaw_occluded:
        # Face turned toward viewer's right -> viewer's right side (matphai 22-29, longmayphai 5-9) is far side
        far_collapsed = (
            abs_yaw >= yaw_outside
            or (eye_r_w < 0.52 * eye_l_w)
            or (local_pts[22, 0] < nose_top_x + 0.12 * eye_l_w)
            or (d_r < 0.26 * max(d_l, 1.0))
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

    elif effective_yaw <= -yaw_occluded:
        # Face turned toward viewer's left -> viewer's left side (mattrai 14-21, longmaytrai 0-4) is far side
        far_collapsed = (
            abs_yaw >= yaw_outside
            or (eye_l_w < 0.52 * eye_r_w)
            or (local_pts[18, 0] > nose_top_x - 0.12 * eye_r_w)
            or (d_l < 0.26 * max(d_r, 1.0))
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

    # 3. Table 3.1 whole-skeleton rules
    table31_borderline: List[str] = []
    for spec in SKELETON_SPECS:
        in_frame_count = sum(1 for pid in spec.point_ids if outside[pid] == 0)
        visible_count = sum(1 for pid in spec.point_ids if outside[pid] == 0 and occluded[pid] == 0)

        if 0 < in_frame_count < spec.min_visible_threshold:
            for pid in spec.point_ids:
                outside[pid] = 1
                occluded[pid] = 0
        elif in_frame_count > 0 and visible_count < spec.min_visible_threshold:
            skel_conf = (
                float(np.mean([point_confidences[pid] for pid in spec.point_ids]))
                if point_confidences is not None
                else 1.0
            )
            if visible_count == 0 and point_confidences is not None and skel_conf < 0.35:
                for pid in spec.point_ids:
                    outside[pid] = 1
                    occluded[pid] = 0
            else:
                table31_borderline.append(f"{spec.name} ({visible_count}/{len(spec.point_ids)})")

    # Coupled lip rule: if outer lip (moingoai) is entirely Outside, inner lip (moitrong) must also be Outside
    if all(outside[pid] == 1 for pid in SKELETON_BY_NAME["moingoai"].point_ids):
        for pid in SKELETON_BY_NAME["moitrong"].point_ids:
            outside[pid] = 1
            occluded[pid] = 0

    # Strict mutual exclusion invariant: Outside=1 => Occluded=0
    for i in range(50):
        if outside[i] == 1:
            occluded[i] = 0

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

    if yaw_geo_conflict or (abs(face_roll_deg) >= 55.0 and abs_yaw >= yaw_occluded):
        reason_cls = (
            f"Góc roll đầu lớn ({face_roll_deg:.1f}°) và góc yaw nghiêng mạnh ({effective_yaw:.1f}°)"
            if not yaw_geo_conflict
            else f"Headpose gốc (yaw={yaw:.1f}°) mâu thuẫn hình học mắt/mũi (hiệu chỉnh yaw={effective_yaw:.1f}°)"
        )
        qc_flags.append(
            QCFlag(
                "UNCERTAIN_CLASS",
                f"{reason_cls}; kiểm tra kỹ bên trái/phải của mattrai/matphai và longmaytrai/longmayphai.",
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
            reason_parts.append(f"góc nghiêng lớn (yaw={effective_yaw:.1f}°)")
        if topology_adjusted:
            reason_parts.append("đã tự nắn topology mí mắt/môi")
        qc_flags.append(
            QCFlag("UNCERTAIN_BOUNDARY", "Cần kiểm tra ranh giới điểm: " + ", ".join(reason_parts) + ".")
        )

    if (yaw_occluded <= abs_yaw < yaw_outside) or table31_borderline or yaw_geo_conflict:
        attr_parts: List[str] = []
        if yaw_occluded <= abs_yaw < yaw_outside:
            attr_parts.append(
                f"Góc yaw={effective_yaw:.1f}° nằm vùng giao thoa; kiểm tra Occluded hay Outside cho mắt/lông mày phía xa."
            )
        if table31_borderline:
            attr_parts.append(
                "Bảng 3.1 (dưới ngưỡng nhìn thấy): "
                + ", ".join(table31_borderline)
                + " — giữ Occluded nếu đủ bằng chứng, không thì gán Outside cả skeleton."
            )
        if yaw_geo_conflict and not attr_parts:
            attr_parts.append("Đã hiệu chỉnh bên Occluded/Outside theo hình học trục mặt; xác nhận lại trạng thái.")
        qc_flags.append(QCFlag("ATTRIBUTE_CHECK", " ".join(attr_parts)))

    return VF50RuleOutput(
        points=annotations,
        qc_flags=qc_flags,
        topology_adjusted=topology_adjusted,
        effective_yaw=effective_yaw,
        face_roll_deg=face_roll_deg,
    )


def revalidate_prediction_qc(
    points: List[PointAnnotation],
    width: int,
    height: int,
    existing_qc_flags: Optional[List[QCFlag]] = None,
) -> List[QCFlag]:
    """Recompute state-dependent QC flags after manual edits while preserving image-context flags."""
    preserved = [
        fl for fl in (existing_qc_flags or [])
        if fl.issue_type in ("LOW_VISIBILITY", "UNCERTAIN_SCOPE", "UNCERTAIN_BOUNDARY", "UNCERTAIN_CLASS")
    ]
    out_flags: List[QCFlag] = []

    # 1. Recompute LOW_CONFIDENCE_POINTS strictly on currently Visible points
    low_conf_pts = [
        p for p in points
        if p.outside == 0 and p.occluded == 0 and p.confidence < 0.60
    ]
    if len(low_conf_pts) >= 2:
        low_conf_pts.sort(key=lambda p: p.confidence)
        top_str = ", ".join(
            f"ID {p.point_id} (±{p.uncertainty_px:.1f}px)" for p in low_conf_pts[:8]
        )
        out_flags.append(
            QCFlag(
                "LOW_CONFIDENCE_POINTS",
                f"Độ đồng thuận model/TTA thấp tại {len(low_conf_pts)} điểm Visible ({top_str}); nên kiểm tra lại.",
            )
        )

    out_flags.extend(preserved)

    # 2. Validate state invariants & Table 3.1 per skeleton
    pts_by_id: Dict[int, PointAnnotation] = {p.point_id: p for p in points}
    attr_issues: List[str] = []

    oob_active = [
        p.point_id for p in points
        if p.outside == 0 and (p.x < 0.0 or p.x >= float(width) or p.y < 0.0 or p.y >= float(height))
    ]
    if oob_active:
        id_preview = ", ".join(str(pid) for pid in oob_active[:6])
        more = "..." if len(oob_active) > 6 else ""
        attr_issues.append(f"Điểm ngoài biên ảnh phải đặt Outside=1 (ID {id_preview}{more}).")

    for spec in SKELETON_SPECS:
        sk_pts = [pts_by_id[pid] for pid in spec.point_ids if pid in pts_by_id]
        if not sk_pts:
            continue
        vis_cnt = sum(1 for p in sk_pts if p.outside == 0 and p.occluded == 0)
        in_frame_cnt = sum(1 for p in sk_pts if p.outside == 0)
        in_bounds_out_cnt = sum(
            1 for p in sk_pts
            if p.outside == 1 and 0.0 <= p.x < float(width) and 0.0 <= p.y < float(height)
        )
        if in_frame_cnt > 0 and vis_cnt < spec.min_visible_threshold:
            attr_issues.append(
                f"{spec.name}: chỉ thấy {vis_cnt}/{len(sk_pts)} (<{spec.min_visible_threshold} điểm Bảng 3.1) — giữ Occluded nếu đủ bằng chứng hoặc gán Outside cả skeleton."
            )
        elif vis_cnt >= spec.min_visible_threshold and in_bounds_out_cnt > 0:
            attr_issues.append(
                f"{spec.name}: đủ ngưỡng ({vis_cnt}/{len(sk_pts)}), {in_bounds_out_cnt} điểm trong ảnh đang Outside nên cân nhắc đổi sang Occluded."
            )

    # Coupled lip validation
    outer_pts = [pts_by_id[pid] for pid in SKELETON_BY_NAME["moingoai"].point_ids if pid in pts_by_id]
    inner_pts = [pts_by_id[pid] for pid in SKELETON_BY_NAME["moitrong"].point_ids if pid in pts_by_id]
    if outer_pts and inner_pts:
        outer_all_out = all(p.outside == 1 for p in outer_pts)
        inner_all_out = all(p.outside == 1 for p in inner_pts)
        outer_vis_cnt = sum(1 for p in outer_pts if p.outside == 0 and p.occluded == 0)
        if outer_all_out and not inner_all_out:
            attr_issues.append("moingoai đã Outside toàn bộ thì moitrong cũng phải gán Outside=1.")
        elif inner_all_out and outer_vis_cnt >= SKELETON_BY_NAME["moingoai"].min_visible_threshold:
            attr_issues.append("moitrong không nên Outside khi moingoai còn nhìn rõ (miệng đóng vẫn giữ 8 điểm moitrong).")

    if attr_issues:
        out_flags.append(QCFlag("ATTRIBUTE_CHECK", " | ".join(attr_issues)))

    return out_flags
