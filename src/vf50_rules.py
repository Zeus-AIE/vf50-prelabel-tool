"""VF-50 annotation rules engine: states (Visible/Occluded/Outside), topology, Table 3.1, and QC flags."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .face_3d_occluder import Face3DEvidence, analyze_3d_occlusion
from .vf50_schema import SKELETON_BY_NAME, SKELETON_SPECS, point_id_to_skeleton

if TYPE_CHECKING:
    from .face_parser import FaceParsingEvidence

VALID_VISIBILITY_MODES = ("auto", "manual")


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
    rule_code: str = "R_PT_VISIBLE"
    reason: str = ""

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
    visibility_mode_used: str = "auto"
    visibility_summary: str = ""
    face_3d_evidence: Optional[Face3DEvidence] = None


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
    pitch: float = 0.0,
    visibility_mode: str = "auto",
    parsing_evidence: Optional["FaceParsingEvidence"] = None,
) -> VF50RuleOutput:
    """Apply VF-50 topology, roll-invariant headpose/auto-visibility states, Table 3.1, and QC flags."""
    if visibility_mode not in VALID_VISIBILITY_MODES:
        raise ValueError(f"Unsupported visibility_mode: {visibility_mode!r}")
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
    eye_ratio_min = min(eye_l_w, eye_r_w) / max(eye_l_w, eye_r_w, 1.0)
    dist_ratio_min = min(d_l, d_r) / max(d_l, d_r, 1.0)
    if abs(geo_score) >= 0.22 and (eye_asym * d_asym > 0.0):
        if effective_yaw * geo_score < 0.0:
            yaw_geo_conflict = True
            effective_yaw = geo_yaw
        elif (
            (abs(geo_score) >= 0.56 and (eye_ratio_min < 0.42 or dist_ratio_min < 0.24))
            or (abs(effective_yaw) < 22.0 and abs(geo_score) >= 0.45 and eye_ratio_min < 0.52)
        ):
            effective_yaw = math.copysign(max(abs(effective_yaw), abs(geo_yaw)), geo_score)

    # Layer 2 prerequisite: 2D-to-3D personalized facial mesh & nasal raycasting (CPU, 0 VRAM)
    ev_3d = analyze_3d_occlusion(local_pts, yaw_deg=effective_yaw, pitch_deg=pitch)

    occluded = np.zeros(50, dtype=int)
    outside = np.zeros(50, dtype=int)
    rule_codes: List[str] = ["R_PT_VISIBLE"] * 50
    reasons: List[str] = [""] * 50
    qc_flags: List[QCFlag] = []
    clear_glasses_eyes = set()

    def _mark_occluded(pid: int, code: str, msg: str, *, overwrite_reason: bool = False) -> None:
        if outside[pid] == 1:
            return
        if occluded[pid] == 0 or overwrite_reason:
            rule_codes[pid] = code
            reasons[pid] = msg
        occluded[pid] = 1

    def _mark_outside(pid: int, code: str, msg: str, *, overwrite_reason: bool = True) -> None:
        if outside[pid] == 1 and rule_codes[pid] == "R_PT_OUT_OF_FRAME":
            return
        if outside[pid] == 0 or overwrite_reason:
            rule_codes[pid] = code
            reasons[pid] = msg
        outside[pid] = 1
        occluded[pid] = 0

    # 1. Out-of-bounds check (outside CVAT canvas MUST be Outside=1, Occluded=0)
    for i in range(50):
        x, y = float(clean_pts[i, 0]), float(clean_pts[i, 1])
        if x < 0.0 or x >= float(img_w) or y < 0.0 or y >= float(img_h):
            _mark_outside(
                i,
                "R_PT_OUT_OF_FRAME",
                f"Tọa độ điểm ({x:.1f}, {y:.1f}) nằm ngoài khung ảnh {img_w}x{img_h}px -> Outside (Mục 3).",
            )

    # 2. Visibility assignment: Auto (BiSeNet + Geometry + 3-Model Consensus) or Manual Yaw Sliders
    abs_yaw = abs(effective_yaw)
    vis_tags: List[str] = []
    auto_attr_notes: List[str] = []
    yaw_borderline_zone = yaw_occluded <= abs_yaw < yaw_outside

    if visibility_mode == "manual":
        visibility_mode_used = "manual"
        visibility_summary = f"manual({yaw_occluded:.0f}°/{yaw_outside:.0f}°)"
        if effective_yaw >= yaw_occluded:
            far_collapsed = (
                abs_yaw >= yaw_outside
                or (eye_r_w < 0.36 * eye_l_w)
                or (local_pts[22, 0] < nose_top_x + 0.12 * eye_l_w)
                or (d_r < 0.24 * max(d_l, 1.0))
                or (eye_r_w < 0.45 * eye_l_w and d_r < 0.26 * max(d_l, 1.0))
            )
            far_ids = list(SKELETON_BY_NAME["longmayphai"].point_ids) + list(SKELETON_BY_NAME["matphai"].point_ids)
            if far_collapsed:
                for pid in far_ids:
                    if outside[pid] == 0:
                        _mark_outside(
                            pid,
                            "R_PT_SELF_OUT_EXTREME_PROFILE",
                            f"Manual Yaw (|yaw|={abs_yaw:.1f}° >= {yaw_outside:.0f}° / dẹt hình học eye_w={eye_r_w:.1f} vs {eye_l_w:.1f}) -> Outside vùng xa phải.",
                            overwrite_reason=False,
                        )
            else:
                for pid in far_ids:
                    _mark_occluded(
                        pid,
                        "R_PT_SELF_OCC_3D_HORIZON",
                        f"Manual Yaw (yaw={effective_yaw:+.1f}° >= {yaw_occluded:.0f}°, cos3D={ev_3d.point_cos_view[pid]:+.2f}) -> Occluded vùng xa phải.",
                    )

            for pid in (10, 35, 36, 37, 46):
                _mark_occluded(
                    pid,
                    "R_PT_SELF_OCC_3D_NOSE" if pid == 10 else "R_PT_SELF_OCC_3D_HORIZON",
                    f"Manual Yaw (yaw={effective_yaw:+.1f}° >= {yaw_occluded:.0f}°) -> Occluded điểm rìa phải.",
                )

        elif effective_yaw <= -yaw_occluded:
            far_collapsed = (
                abs_yaw >= yaw_outside
                or (eye_l_w < 0.36 * eye_r_w)
                or (local_pts[18, 0] > nose_top_x - 0.12 * eye_r_w)
                or (d_l < 0.24 * max(d_r, 1.0))
                or (eye_l_w < 0.45 * eye_r_w and d_l < 0.26 * max(d_r, 1.0))
            )
            far_ids = list(SKELETON_BY_NAME["longmaytrai"].point_ids) + list(SKELETON_BY_NAME["mattrai"].point_ids)
            if far_collapsed:
                for pid in far_ids:
                    if outside[pid] == 0:
                        _mark_outside(
                            pid,
                            "R_PT_SELF_OUT_EXTREME_PROFILE",
                            f"Manual Yaw (|yaw|={abs_yaw:.1f}° >= {yaw_outside:.0f}° / dẹt hình học eye_w={eye_l_w:.1f} vs {eye_r_w:.1f}) -> Outside vùng xa trái.",
                            overwrite_reason=False,
                        )
            else:
                for pid in far_ids:
                    _mark_occluded(
                        pid,
                        "R_PT_SELF_OCC_3D_HORIZON",
                        f"Manual Yaw (yaw={effective_yaw:+.1f}° <= -{yaw_occluded:.0f}°, cos3D={ev_3d.point_cos_view[pid]:+.2f}) -> Occluded vùng xa trái.",
                    )

            for pid in (10, 30, 31, 41, 42):
                _mark_occluded(
                    pid,
                    "R_PT_SELF_OCC_3D_NOSE" if pid == 10 else "R_PT_SELF_OCC_3D_HORIZON",
                    f"Manual Yaw (yaw={effective_yaw:+.1f}° <= -{yaw_occluded:.0f}°) -> Occluded điểm rìa trái.",
                )
    else:
        visibility_mode_used = "auto"
        has_valid_parser = bool(parsing_evidence is not None and parsing_evidence.valid)
        wearing_glasses_any = bool(
            has_valid_parser
            and parsing_evidence is not None
            and (
                parsing_evidence.eye_l_eyeg_prob >= 0.35
                or parsing_evidence.eye_r_eyeg_prob >= 0.35
            )
        )

        # A. Auto far-side self-occlusion (geometry + pitch-adjusted yaw + 3-model confidence + BiSeNet)
        yaw_eff_auto = abs_yaw + 0.15 * max(abs(pitch) - 15.0, 0.0)
        if effective_yaw >= 0.0:
            profile_tag = "profile_R"
            far_brow_ids = list(SKELETON_BY_NAME["longmayphai"].point_ids)
            far_eye_ids = list(SKELETON_BY_NAME["matphai"].point_ids)
            far_corner_ids = (10, 35, 36, 37, 46)
            eye_near_w, eye_far_w = eye_l_w, eye_r_w
            d_near, d_far = d_l, d_r
            nose_bridge_overlap = bool(local_pts[22, 0] < nose_top_x + 0.12 * eye_l_w)
            near_eye_slice, far_eye_slice = slice(14, 22), slice(22, 30)
            far_brow_slice = slice(5, 10)
            far_eye_key = "r"
        else:
            profile_tag = "profile_L"
            far_brow_ids = list(SKELETON_BY_NAME["longmaytrai"].point_ids)
            far_eye_ids = list(SKELETON_BY_NAME["mattrai"].point_ids)
            far_corner_ids = (10, 30, 31, 41, 42)
            eye_near_w, eye_far_w = eye_r_w, eye_l_w
            d_near, d_far = d_r, d_l
            nose_bridge_overlap = bool(local_pts[18, 0] > nose_top_x - 0.12 * eye_r_w)
            near_eye_slice, far_eye_slice = slice(22, 30), slice(14, 22)
            far_brow_slice = slice(0, 5)
            far_eye_key = "l"

        eye_ratio = eye_far_w / max(eye_near_w, 1.0)
        dist_ratio = d_far / max(d_near, 1.0)
        if point_confidences is not None:
            conf_near_eye = float(np.mean(point_confidences[near_eye_slice]))
            conf_far_eye = float(np.mean(point_confidences[far_eye_slice]))
            conf_far_brow = float(np.mean(point_confidences[far_brow_slice]))
            conf_drop = max(0.0, conf_near_eye - conf_far_eye)
        else:
            conf_near_eye = conf_far_eye = conf_far_brow = 1.0
            conf_drop = 0.0

        if has_valid_parser and parsing_evidence is not None:
            far_eye_vis = parsing_evidence.eye_r_vis_prob if far_eye_key == "r" else parsing_evidence.eye_l_vis_prob
            near_eye_vis = parsing_evidence.eye_l_vis_prob if far_eye_key == "r" else parsing_evidence.eye_r_vis_prob
            far_eye_eyeg = parsing_evidence.eye_r_eyeg_prob if far_eye_key == "r" else parsing_evidence.eye_l_eyeg_prob
            far_eye_nose = parsing_evidence.eye_r_nose_prob if far_eye_key == "r" else parsing_evidence.eye_l_nose_prob
            far_eye_bg = parsing_evidence.eye_r_bg_prob if far_eye_key == "r" else parsing_evidence.eye_l_bg_prob
            far_brow_exp = float(np.mean(parsing_evidence.point_expected[far_brow_slice]))
            far_brow_bg = float(np.mean(parsing_evidence.point_bg[far_brow_slice]))
            far_brow_hair = float(np.mean(parsing_evidence.point_hair_hat[far_brow_slice]))
            far_brow_vis_pts = int(
                np.sum(
                    (parsing_evidence.point_expected[far_brow_slice] >= 0.20)
                    & (parsing_evidence.point_bg[far_brow_slice] < 0.25)
                    & (parsing_evidence.point_hair_hat[far_brow_slice] < 0.50)
                )
            )
        else:
            far_eye_vis = near_eye_vis = far_eye_eyeg = far_eye_nose = far_eye_bg = 0.0
            far_brow_exp = far_brow_bg = far_brow_hair = 0.0
            far_brow_vis_pts = 0

        geo_collapsed = (
            (eye_ratio < 0.36)
            or (dist_ratio < 0.24)
            or nose_bridge_overlap
            or (eye_ratio < 0.45 and dist_ratio < 0.26)
            or (ev_3d.far_eye_ridge_cross_3d >= 3)
        )
        parser_early_collapsed = (yaw_eff_auto >= 45.0) and (
            (
                has_valid_parser
                and far_eye_vis < 0.12
                and far_eye_eyeg < 0.25
                and (far_eye_nose + far_eye_bg) >= 0.35
            )
            or conf_drop >= 0.28
        )
        far_eye_vis_rel = far_eye_vis / max(near_eye_vis, 0.15)
        strong_far_visible_guard = (
            (yaw_eff_auto < 62.0)
            and (eye_ratio >= 0.42)
            and not nose_bridge_overlap
            and has_valid_parser
            and (far_eye_nose < 0.22)
            and (far_eye_bg < 0.15)
            and (conf_far_eye >= 0.72)
            and (
                far_eye_vis >= 0.35
                or (far_eye_vis >= 0.15 and far_eye_vis_rel >= 0.55)
            )
        )
        horizon_3d_collapsed = (
            (abs_yaw >= 48.0)
            and (ev_3d.far_eye_cos_mean < 0.10)
            and (ev_3d.far_eye_cos_min < -0.10)
            and not strong_far_visible_guard
        )
        far_eye_clearly_visible = (
            has_valid_parser
            and (yaw_eff_auto < 48.0)
            and (ev_3d.far_eye_cos_mean >= 0.30)
            and (ev_3d.far_eye_cos_min >= 0.15)
            and (ev_3d.far_eye_nose_hits_3d == 0)
            and not nose_bridge_overlap
            and (far_eye_nose < 0.22)
            and (far_eye_bg < 0.15)
            and (conf_far_eye >= 0.72)
            and (
                far_eye_vis >= 0.30
                or (far_eye_vis >= 0.14 and far_eye_vis_rel >= 0.55)
            )
        )
        far_eye_collapsed = geo_collapsed or parser_early_collapsed or horizon_3d_collapsed or (
            (abs_yaw >= 18.0) and (yaw_eff_auto >= 55.0) and not strong_far_visible_guard
        )
        far_eye_occluded = (not far_eye_collapsed) and (not far_eye_clearly_visible) and (
            ((abs_yaw >= 18.0) and (yaw_eff_auto >= 35.0))
            or ((abs_yaw >= 18.0) and (ev_3d.far_eye_nose_hits_3d >= 1 or ev_3d.far_eye_cos_min < 0.05))
            or (
                (abs_yaw >= 18.0)
                and (yaw_eff_auto >= 25.0)
                and (
                    ((not wearing_glasses_any) and (eye_ratio < 0.72))
                    or dist_ratio < 0.52
                    or (has_valid_parser and far_eye_nose >= 0.25)
                    or conf_drop >= 0.20
                )
            )
        )

        # Eyebrow sits above the nose bridge: evaluate 3D normal + BiSeNet independently
        far_brow_clearly_visible = (
            has_valid_parser
            and (yaw_eff_auto < 50.0)
            and (ev_3d.far_brow_cos_mean >= 0.25)
            and (ev_3d.far_brow_cos_min >= 0.0)
            and (far_brow_exp >= 0.28)
            and (far_brow_vis_pts >= 3)
            and (far_brow_bg < 0.18)
            and (far_brow_hair < 0.25)
            and (conf_far_brow >= 0.75)
        )
        far_brow_protected_from_outside = (
            has_valid_parser
            and (yaw_eff_auto < 60.0)
            and (far_brow_exp >= 0.22)
            and (far_brow_vis_pts >= 3)
            and (far_brow_bg < 0.20)
            and (conf_far_brow >= 0.72)
        )
        far_brow_collapsed = far_eye_collapsed and not far_brow_protected_from_outside
        far_brow_occluded = (
            (not far_brow_collapsed)
            and (far_eye_occluded or (abs_yaw >= 25.0 and ev_3d.far_brow_cos_min < -0.10))
            and (not far_brow_clearly_visible)
        )

        far_corner_occluded = far_eye_collapsed or (
            far_eye_occluded
            and (
                (yaw_eff_auto >= 42.0)
                or (abs_yaw >= 38.0 and ev_3d.far_corner_cos_mean < 0.35)
            )
        )

        yaw_borderline_zone = (not far_eye_collapsed) and (far_eye_occluded or far_brow_occluded)
        if far_eye_collapsed or far_eye_occluded or far_brow_collapsed or far_brow_occluded:
            vis_tags.append(profile_tag)

        if far_eye_collapsed:
            for pid in far_eye_ids:
                _mark_outside(
                    pid,
                    "R_PT_SELF_OUT_EXTREME_PROFILE",
                    f"Góc nghiêng cực đại (yaw={effective_yaw:+.1f}°, cos3D={ev_3d.point_cos_view[pid]:+.2f}, eye_ratio={eye_ratio:.2f}) -> vùng mắt xa khuất hoàn toàn (Outside).",
                    overwrite_reason=False,
                )
        elif far_eye_occluded:
            for pid in far_eye_ids:
                if ev_3d.point_nose_occluded[pid]:
                    _mark_occluded(
                        pid,
                        "R_PT_SELF_OCC_3D_NOSE",
                        f"Tháp mũi 3D cản tia nhìn tại mắt xa (nose_hits={ev_3d.far_eye_nose_hits_3d}, depth={ev_3d.point_nose_ray_depth[pid]:.2f}, yaw={effective_yaw:+.1f}°) -> Occluded.",
                    )
                else:
                    _mark_occluded(
                        pid,
                        "R_PT_SELF_OCC_3D_HORIZON",
                        f"Góc nghiêng tự che vùng mắt xa (yaw={effective_yaw:+.1f}°, cos3D={ev_3d.point_cos_view[pid]:+.2f}, eye_vis={far_eye_vis:.2f}) -> Occluded.",
                    )

        if (
            (not far_eye_collapsed)
            and (not far_eye_occluded)
            and (abs_yaw >= 18.0)
            and has_valid_parser
            and parsing_evidence is not None
        ):
            for pid in far_eye_ids:
                if ev_3d.point_nose_occluded[pid] and parsing_evidence.point_nose[pid] >= 0.35:
                    _mark_occluded(
                        pid,
                        "R_PT_SELF_OCC_3D_NOSE",
                        f"Tháp mũi 3D che cục bộ góc mắt xa (depth={ev_3d.point_nose_ray_depth[pid]:.2f}, nose_prob={parsing_evidence.point_nose[pid]:.2f}) -> Occluded.",
                    )

        if far_brow_collapsed:
            for pid in far_brow_ids:
                _mark_outside(
                    pid,
                    "R_PT_SELF_OUT_EXTREME_PROFILE",
                    f"Góc nghiêng cực đại (yaw={effective_yaw:+.1f}°, cos3D={ev_3d.point_cos_view[pid]:+.2f}) -> lông mày xa khuất hoàn toàn (Outside).",
                    overwrite_reason=False,
                )
        elif far_brow_occluded:
            for pid in far_brow_ids:
                _mark_occluded(
                    pid,
                    "R_PT_SELF_OCC_3D_HORIZON",
                    f"Lông mày phía xa nghiêng ra rìa mặt (yaw={effective_yaw:+.1f}°, cos3D={ev_3d.point_cos_view[pid]:+.2f}) -> Occluded.",
                )

        if far_corner_occluded:
            for pid in far_corner_ids:
                _mark_occluded(
                    pid,
                    "R_PT_SELF_OCC_3D_NOSE" if pid == 10 else "R_PT_SELF_OCC_3D_HORIZON",
                    f"Góc nghiêng sâu (yaw={effective_yaw:+.1f}°, corner_cos={ev_3d.far_corner_cos_mean:+.2f}) khiến điểm rìa sống mũi/khóe môi xa bị che -> Occluded.",
                )

        # B. Auto external occluders via BiSeNet Face Parsing (even at Yaw = 0°)
        if has_valid_parser and parsing_evidence is not None:
            cheek_ctr = parsing_evidence.cheek_contrast
            e_l_vis = parsing_evidence.eye_l_vis_prob
            e_r_vis = parsing_evidence.eye_r_vis_prob
            e_l_eyeg = parsing_evidence.eye_l_eyeg_prob
            e_r_eyeg = parsing_evidence.eye_r_eyeg_prob
            e_l_ctr = parsing_evidence.eye_l_contrast
            e_r_ctr = parsing_evidence.eye_r_contrast
            both_eyeg = (e_l_eyeg >= 0.45) and (e_r_eyeg >= 0.45)
            max_eye_vis = max(e_l_vis, e_r_vis)
            min_eye_ctr = min(e_l_ctr, e_r_ctr)
            paired_black_sunglasses = (
                both_eyeg
                and (max_eye_vis < 0.022)
                and (min_eye_ctr < 9.5)
                and (min_eye_ctr < 0.38 * max(cheek_ctr, 12.0))
            )

            # 1) Opaque sunglasses / IR glare vs clear glasses per eye
            eye_configs = (
                (
                    "mattrai",
                    range(14, 22),
                    e_l_vis,
                    e_l_eyeg,
                    e_l_ctr,
                    parsing_evidence.eye_l_glare_ratio,
                ),
                (
                    "matphai",
                    range(22, 30),
                    e_r_vis,
                    e_r_eyeg,
                    e_r_ctr,
                    parsing_evidence.eye_r_glare_ratio,
                ),
            )
            for eye_name, pids, eye_vis, eye_eyeg, eye_ctr, eye_glare in eye_configs:
                if all(outside[pid] == 1 for pid in pids):
                    continue
                conf_eye = (
                    float(np.mean([point_confidences[pid] for pid in pids]))
                    if point_confidences is not None
                    else 1.0
                )
                is_glare = (
                    (not paired_black_sunglasses)
                    and (eye_glare >= 0.40)
                    and (eye_vis < 0.20)
                    and (eye_ctr < 10.0 or eye_glare >= 0.65)
                )
                is_opaque_glasses = (
                    (eye_eyeg >= 0.35)
                    and (eye_vis < 0.12)
                    and (
                        (eye_ctr < 8.5 and eye_ctr < 0.38 * max(cheek_ctr, 12.0))
                        or (point_confidences is not None and conf_eye < 0.58)
                    )
                )
                is_clear_glasses = (
                    (not paired_black_sunglasses)
                    and (eye_eyeg >= 0.35)
                    and not is_opaque_glasses
                    and not is_glare
                )
                if is_clear_glasses:
                    clear_glasses_eyes.add(eye_name)
                    vis_tags.append("glasses_clear")
                elif paired_black_sunglasses or is_glare or is_opaque_glasses:
                    collapse_eye = paired_black_sunglasses or (
                        eye_vis < 0.05
                        and point_confidences is not None
                        and conf_eye < 0.42
                        and (eye_glare >= 0.70 or eye_ctr < 4.0)
                    )
                    for pid in pids:
                        if outside[pid] == 1 and rule_codes[pid] == "R_PT_OUT_OF_FRAME":
                            continue
                        if collapse_eye:
                            sg_code = "R_PT_SUNGLASSES_BLACK_OUT"
                            sg_msg = (
                                f"Kính râm đen che kín cả 2 mắt (vis_max={max_eye_vis:.3f}, ctr_min={min_eye_ctr:.1f}, "
                                f"eyeg={e_l_eyeg:.2f}/{e_r_eyeg:.2f}) -> Outside theo Mục 4.3b."
                                if paired_black_sunglasses
                                else f"Kính tối/che kín mất dấu mắt {eye_name} (vis={eye_vis:.3f}, ctr={eye_ctr:.1f}, conf={conf_eye:.2f}) -> Outside (Mục 4.3b)."
                            )
                            _mark_outside(pid, sg_code, sg_msg, overwrite_reason=True)
                        elif outside[pid] == 0:
                            if is_glare:
                                _mark_occluded(
                                    pid,
                                    "R_PT_GLASSES_GLARE_OCC",
                                    f"Lóa sáng/phản quang IR che tròng kính {eye_name} (glare={eye_glare:.2f}, vis={eye_vis:.2f}, ctr={eye_ctr:.1f}) -> Occluded (Mục 4.3d).",
                                    overwrite_reason=True,
                                )
                            else:
                                _mark_occluded(
                                    pid,
                                    "R_PT_GLASSES_FRAME_RIM",
                                    f"Tròng kính tối / gọng dày che mờ {eye_name} (eyeg={eye_eyeg:.2f}, vis={eye_vis:.2f}, ctr={eye_ctr:.1f}) -> Occluded.",
                                    overwrite_reason=True,
                                )
                    if not paired_black_sunglasses:
                        if is_glare:
                            vis_tags.append("ir_glare")
                            auto_attr_notes.append(f"BiSeNet phát hiện lóa sáng IR che tròng mắt ({eye_name}).")
                        else:
                            vis_tags.append("glasses_opaque")
                            auto_attr_notes.append(f"BiSeNet phát hiện kính râm tối/che mờ mắt ({eye_name}).")

            if paired_black_sunglasses:
                for pid in range(0, 10):
                    if outside[pid] == 0 and (
                        parsing_evidence.point_eyeg[pid] >= 0.30
                        and parsing_evidence.point_expected[pid] < 0.25
                    ):
                        _mark_occluded(
                            pid,
                            "R_PT_GLASSES_FRAME_RIM",
                            f"Viền/gọng trên kính râm che đè lên điểm lông mày (point_eyeg={parsing_evidence.point_eyeg[pid]:.2f}, brow_vis={parsing_evidence.point_expected[pid]:.2f}) -> Occluded (Mục 4.3c).",
                            overwrite_reason=True,
                        )
                vis_tags.append("glasses_opaque")
                auto_attr_notes.append(
                    f"Phát hiện kính râm đen che kín cả 2 mắt (vis_max={max_eye_vis:.3f}, ctr_min={min_eye_ctr:.1f}): "
                    "mattrai+matphai -> Outside (Mục 4.3b), lông mày dưới viền gọng -> Occluded."
                )
                if abs_yaw < 35.0:
                    vis_tags = [t for t in vis_tags if t != profile_tag]
                    yaw_borderline_zone = False

            # 2) Face mask / cloth / hand covering mouth & lower nose
            mask_pt_cnt = 0
            for pid in range(30, 50):
                pt_masked = (
                    parsing_evidence.point_cloth_bg[pid] >= 0.50
                    and parsing_evidence.point_expected[pid] < 0.10
                    and parsing_evidence.point_skin[pid] < 0.30
                )
                if pt_masked:
                    if pid < 42:
                        mask_pt_cnt += 1
                    _mark_occluded(
                        pid,
                        "R_PT_MASK_OCC",
                        f"Khẩu trang/vải che lên điểm môi (cloth_bg={parsing_evidence.point_cloth_bg[pid]:.2f}, lip_vis={parsing_evidence.point_expected[pid]:.2f}) -> Occluded (Mục 4.4a).",
                    )

            colored_mask = (parsing_evidence.lip_vis_prob < 0.06) and (
                parsing_evidence.lip_cloth_prob >= 0.22
                or parsing_evidence.lip_cloth_bg_prob >= 0.40
                or mask_pt_cnt >= 6
            )
            mono_flat_mask = (
                (parsing_evidence.lip_vis_prob < 0.04)
                and (parsing_evidence.lip_contrast < 6.0)
                and (parsing_evidence.lip_cloth_bg_prob >= 0.08 or cheek_ctr >= 12.0)
            )
            if colored_mask or mono_flat_mask:
                conf_mouth = (
                    float(np.mean(point_confidences[30:42]))
                    if point_confidences is not None
                    else 1.0
                )
                collapse_mouth = point_confidences is not None and conf_mouth < 0.45
                for pid in range(30, 50):
                    if outside[pid] == 1 and rule_codes[pid] == "R_PT_OUT_OF_FRAME":
                        continue
                    if collapse_mouth:
                        _mark_outside(
                            pid,
                            "R_PT_MASK_OUT",
                            f"Khẩu trang/vật cản che kín toàn bộ vùng miệng (lip_vis={parsing_evidence.lip_vis_prob:.2f}, lip_ctr={parsing_evidence.lip_contrast:.1f}, conf={conf_mouth:.2f}) -> Outside (Mục 4.4b).",
                            overwrite_reason=True,
                        )
                    else:
                        _mark_occluded(
                            pid,
                            "R_PT_MASK_OCC",
                            f"Khẩu trang/vải che phủ vùng miệng (lip_vis={parsing_evidence.lip_vis_prob:.2f}, lip_cloth={parsing_evidence.lip_cloth_prob:.2f}) -> Occluded (Mục 4.4a).",
                        )
                for nose_pid in (12, 13):
                    if (
                        (parsing_evidence.point_cloth_bg[nose_pid] >= 0.50 and parsing_evidence.point_nose[nose_pid] < 0.20)
                        or parsing_evidence.point_nose[nose_pid] < 0.15
                    ):
                        _mark_occluded(
                            nose_pid,
                            "R_PT_MASK_OCC",
                            f"Khẩu trang/vật cản che lên chóp/gốc mũi dưới (cloth_bg={parsing_evidence.point_cloth_bg[nose_pid]:.2f}, nose_prob={parsing_evidence.point_nose[nose_pid]:.2f}) -> Occluded.",
                        )
                vis_tags.append("mask_mouth")
                auto_attr_notes.append("BiSeNet phát hiện khẩu trang / vật cản che vùng miệng (moingoai/moitrong).")

            # 3) Hair / Hat covering eyebrows (0..9) & eyes (14..29)
            hair_hit = False
            for grp_pids in (range(0, 5), range(5, 10), range(14, 22), range(22, 30)):
                covered_cnt = 0
                for pid in grp_pids:
                    if parsing_evidence.point_hair_hat[pid] >= 0.75 and parsing_evidence.point_expected[pid] < 0.12:
                        covered_cnt += 1
                        hair_hit = True
                        _mark_occluded(
                            pid,
                            "R_PT_HAIR_OCC",
                            f"Tóc mái/mũ che cục bộ lên điểm (hair_hat={parsing_evidence.point_hair_hat[pid]:.2f}, vis={parsing_evidence.point_expected[pid]:.2f}) -> Occluded.",
                        )
                if covered_cnt >= int(math.ceil(0.8 * len(grp_pids))) and point_confidences is not None:
                    grp_conf = float(np.mean([point_confidences[pid] for pid in grp_pids]))
                    if grp_conf < 0.45:
                        for pid in grp_pids:
                            if outside[pid] == 1 and rule_codes[pid] == "R_PT_OUT_OF_FRAME":
                                continue
                            _mark_outside(
                                pid,
                                "R_PT_HAIR_OUT",
                                f"Tóc/vật cản che kín toàn bộ nhóm ({covered_cnt}/{len(grp_pids)} điểm, conf={grp_conf:.2f}) -> Outside (Bảng 3.1).",
                                overwrite_reason=True,
                            )
            if hair_hit:
                vis_tags.append("hair_occ")
                auto_attr_notes.append("BiSeNet phát hiện tóc mái / mũ che một phần lông mày hoặc mắt.")

        base_tag = "auto(3d+bisenet+geo)" if has_valid_parser else "auto(geo+3d)"
        uniq_tags = list(dict.fromkeys(vis_tags))
        visibility_summary = f"{base_tag[:-1]}: {', '.join(uniq_tags)})" if uniq_tags else base_tag

    # 3. Table 3.1 whole-skeleton rules
    table31_borderline: List[str] = []
    for spec in SKELETON_SPECS:
        in_frame_count = sum(1 for pid in spec.point_ids if outside[pid] == 0)
        visible_count = sum(1 for pid in spec.point_ids if outside[pid] == 0 and occluded[pid] == 0)

        if 0 < in_frame_count < spec.min_visible_threshold:
            for pid in spec.point_ids:
                if outside[pid] == 0:
                    _mark_outside(
                        pid,
                        "R_SKEL_TABLE31_COLLAPSE",
                        f"Sập skeleton {spec.name} theo Bảng 3.1 (chỉ còn {in_frame_count}/{len(spec.point_ids)} trong khung < {spec.min_visible_threshold}) -> toàn bộ Outside.",
                        overwrite_reason=False,
                    )
        elif in_frame_count > 0 and visible_count < spec.min_visible_threshold:
            skel_conf = (
                float(np.mean([point_confidences[pid] for pid in spec.point_ids]))
                if point_confidences is not None
                else 1.0
            )
            if visible_count == 0 and point_confidences is not None and skel_conf < 0.35:
                for pid in spec.point_ids:
                    if outside[pid] == 0:
                        _mark_outside(
                            pid,
                            "R_SKEL_TABLE31_COLLAPSE",
                            f"Sập skeleton {spec.name} theo Bảng 3.1 (0/{len(spec.point_ids)} Visible, conf={skel_conf:.2f} < 0.35) -> toàn bộ Outside.",
                            overwrite_reason=False,
                        )
            else:
                table31_borderline.append(f"{spec.name} ({visible_count}/{len(spec.point_ids)})")

    # Coupled lip rule: if outer lip (moingoai) is entirely Outside, inner lip (moitrong) must also be Outside
    if all(outside[pid] == 1 for pid in SKELETON_BY_NAME["moingoai"].point_ids):
        for pid in SKELETON_BY_NAME["moitrong"].point_ids:
            if outside[pid] == 0:
                _mark_outside(
                    pid,
                    "R_SKEL_COUPLED_LIPS",
                    "Đồng bộ moitrong Outside theo Bảng 3.1 do moingoai đã sập toàn bộ Outside.",
                    overwrite_reason=False,
                )

    # Strict mutual exclusion invariant: Outside=1 => Occluded=0
    for i in range(50):
        if outside[i] == 1:
            occluded[i] = 0

    # Assign final Guideline rule_code & reason for remaining Visible points
    eye_l_h = float(np.mean([
        abs(float(local_pts[15, 1] - local_pts[21, 1])),
        abs(float(local_pts[16, 1] - local_pts[20, 1])),
        abs(float(local_pts[17, 1] - local_pts[19, 1])),
    ]))
    eye_r_h = float(np.mean([
        abs(float(local_pts[23, 1] - local_pts[29, 1])),
        abs(float(local_pts[24, 1] - local_pts[28, 1])),
        abs(float(local_pts[25, 1] - local_pts[27, 1])),
    ]))
    closed_eye_l = (eye_l_h / max(eye_l_w, 1.0)) < 0.16
    closed_eye_r = (eye_r_h / max(eye_r_w, 1.0)) < 0.16
    mouth_w_ref = max(float(np.linalg.norm(clean_pts[36] - clean_pts[30])), 1.0)
    mouth_h_ref = float(np.mean([
        abs(float(local_pts[43, 1] - local_pts[49, 1])),
        abs(float(local_pts[44, 1] - local_pts[48, 1])),
        abs(float(local_pts[45, 1] - local_pts[47, 1])),
    ]))
    closed_mouth = (mouth_h_ref / mouth_w_ref) < 0.08

    for pid in range(50):
        if outside[pid] == 0 and occluded[pid] == 0:
            sk_name = point_id_to_skeleton(pid)
            cos_v = float(ev_3d.point_cos_view[pid])
            if 14 <= pid <= 21:
                if "mattrai" in clear_glasses_eyes and parsing_evidence is not None:
                    rule_codes[pid] = "R_PT_GLASSES_CLEAR"
                    reasons[pid] = (
                        f"Kính trong nhìn xuyên qua rõ mắt trái (eyeg={parsing_evidence.eye_l_eyeg_prob:.2f}, "
                        f"ctr={parsing_evidence.eye_l_contrast:.1f}) -> Visible (Mục 4.3a)."
                    )
                elif closed_eye_l:
                    rule_codes[pid] = "R_PT_CLOSED_EYE_VISIBLE"
                    reasons[pid] = (
                        f"Mắt trái nhắm/nheo (tỷ lệ mở={eye_l_h / max(eye_l_w, 1.0):.2f} < 0.16) nhưng thấy rõ khe mí -> Visible (Mục 4.1)."
                    )
                else:
                    rule_codes[pid] = "R_PT_VISIBLE"
                    reasons[pid] = f"Điểm mattrai nhìn thấy rõ (cos3D={cos_v:+.2f}, yaw={effective_yaw:+.1f}°)."
            elif 22 <= pid <= 29:
                if "matphai" in clear_glasses_eyes and parsing_evidence is not None:
                    rule_codes[pid] = "R_PT_GLASSES_CLEAR"
                    reasons[pid] = (
                        f"Kính trong nhìn xuyên qua rõ mắt phải (eyeg={parsing_evidence.eye_r_eyeg_prob:.2f}, "
                        f"ctr={parsing_evidence.eye_r_contrast:.1f}) -> Visible (Mục 4.3a)."
                    )
                elif closed_eye_r:
                    rule_codes[pid] = "R_PT_CLOSED_EYE_VISIBLE"
                    reasons[pid] = (
                        f"Mắt phải nhắm/nheo (tỷ lệ mở={eye_r_h / max(eye_r_w, 1.0):.2f} < 0.16) nhưng thấy rõ khe mí -> Visible (Mục 4.1)."
                    )
                else:
                    rule_codes[pid] = "R_PT_VISIBLE"
                    reasons[pid] = f"Điểm matphai nhìn thấy rõ (cos3D={cos_v:+.2f}, yaw={effective_yaw:+.1f}°)."
            elif 42 <= pid <= 49 and closed_mouth:
                rule_codes[pid] = "R_PT_CLOSED_MOUTH_VISIBLE"
                reasons[pid] = (
                    f"Miệng ngậm khép (tỷ lệ mở={mouth_h_ref / mouth_w_ref:.2f} < 0.08), 8 điểm moitrong bám đường khép môi -> Visible (Mục 4.2)."
                )
            else:
                rule_codes[pid] = "R_PT_VISIBLE"
                reasons[pid] = f"Điểm {sk_name} nhìn thấy rõ trong khung (cos3D={cos_v:+.2f}, yaw={effective_yaw:+.1f}°)."

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
                rule_code=rule_codes[pid],
                reason=reasons[pid],
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

    if yaw_borderline_zone or table31_borderline or yaw_geo_conflict or auto_attr_notes:
        attr_parts: List[str] = list(auto_attr_notes)
        if yaw_borderline_zone:
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
        visibility_mode_used=visibility_mode_used,
        visibility_summary=visibility_summary,
        face_3d_evidence=ev_3d,
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
