"""2D-to-3D personalized face mesh lifting & nasal pyramid Z-buffer raycasting (CPU, 0 MB VRAM)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

# Canonical 50-vertex 3D facial coordinates derived from SPIGA mean_face_3D_98
# Coordinate frame: +X = viewer right, +Y = viewer down, +Z = toward camera (nose tip 13 at origin)
_CANONICAL_VF50_XYZ = np.array(
    [
        # 0..4 longmaytrai
        [-0.5522, -0.4838, -0.4665],
        [-0.4598, -0.5618, -0.3082],
        [-0.3602, -0.5663, -0.2168],
        [-0.2498, -0.5303, -0.1487],
        [-0.1019, -0.4825, -0.1242],
        # 5..9 longmayphai
        [0.0980, -0.4659, -0.1317],
        [0.2143, -0.4971, -0.1357],
        [0.3730, -0.5204, -0.2230],
        [0.5016, -0.5004, -0.3193],
        [0.5736, -0.4578, -0.4651],
        # 10..13 songmui
        [0.0084, -0.4163, -0.1391],
        [0.0073, -0.2672, -0.1065],
        [0.0019, -0.1424, -0.0545],
        [0.0000, 0.0000, 0.0000],
        # 14..21 mattrai
        [-0.4063, -0.3604, -0.3833],
        [-0.3448, -0.3843, -0.3103],
        [-0.3031, -0.3990, -0.2984],
        [-0.2615, -0.3960, -0.2866],
        [-0.1608, -0.3710, -0.3302],
        [-0.2456, -0.3344, -0.3231],
        [-0.2869, -0.3244, -0.3315],
        [-0.3281, -0.3320, -0.3399],
        # 22..29 matphai
        [0.1639, -0.3407, -0.3141],
        [0.2508, -0.3925, -0.2790],
        [0.2958, -0.3945, -0.2824],
        [0.3408, -0.3790, -0.2858],
        [0.4324, -0.3487, -0.3772],
        [0.3335, -0.3280, -0.3185],
        [0.2900, -0.3227, -0.3158],
        [0.2466, -0.3351, -0.3131],
        # 30..41 moingoai
        [-0.2230, 0.2776, -0.3543],
        [-0.1571, 0.2487, -0.2702],
        [-0.0816, 0.2362, -0.2252],
        [-0.0039, 0.2450, -0.2126],
        [0.0629, 0.2308, -0.2085],
        [0.1355, 0.2406, -0.2355],
        [0.2404, 0.2837, -0.3323],
        [0.1437, 0.3295, -0.2840],
        [0.0841, 0.3554, -0.2565],
        [0.0081, 0.3604, -0.2504],
        [-0.0778, 0.3490, -0.2697],
        [-0.1571, 0.3120, -0.3224],
        # 42..49 moitrong
        [-0.1827, 0.2784, -0.3412],
        [-0.0759, 0.2835, -0.2709],
        [-0.0027, 0.2869, -0.2544],
        [0.0685, 0.2856, -0.2578],
        [0.1942, 0.2796, -0.3146],
        [0.0690, 0.2881, -0.2587],
        [-0.0027, 0.2892, -0.2562],
        [-0.0767, 0.2863, -0.2730],
    ],
    dtype=np.float64,
)

_HEAD_PIVOT = np.array([0.0, 0.0, -0.45], dtype=np.float64)


def _build_canonical_normals() -> np.ndarray:
    """Compute anatomical outward surface normals from semi-cylindrical cranial center."""
    nrm = _CANONICAL_VF50_XYZ - np.array([0.0, 0.05, -0.80], dtype=np.float64)
    nrm[:, 1] *= 0.45
    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm / np.maximum(norms, 1e-8)


_CANONICAL_VF50_NORMALS = _build_canonical_normals()

# 12 triangles on nasal pyramid: 0..3 ridge (10..13), 4..7 left base wall, 8..11 right base wall
_NASAL_PYRAMID_TRIANGLES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 5), (0, 5, 4),
    (1, 2, 6), (1, 6, 5),
    (2, 3, 7), (2, 7, 6),
    (0, 1, 9), (0, 9, 8),
    (1, 2, 10), (1, 10, 9),
    (2, 3, 11), (2, 11, 10),
)

# Anatomical 3D surface triangulation over the 50 VF-50 vertices for interactive rendering
_FACE_MESH_TRIANGLES: Tuple[Tuple[int, int, int], ...] = (
    # Left brow to left upper eyelid
    (0, 1, 14), (1, 15, 14), (1, 2, 15), (2, 16, 15), (2, 3, 16), (3, 17, 16), (3, 4, 17), (4, 18, 17),
    # Glabella & nose root
    (4, 10, 18), (4, 5, 10), (5, 22, 10),
    # Right brow to right upper eyelid
    (5, 6, 22), (6, 23, 22), (6, 7, 23), (7, 24, 23), (7, 8, 24), (8, 25, 24), (8, 9, 25), (9, 26, 25),
    # Left eye socket surface
    (14, 15, 21), (15, 20, 21), (15, 16, 20), (16, 19, 20), (16, 17, 19), (17, 18, 19),
    # Right eye socket surface
    (22, 23, 29), (23, 28, 29), (23, 24, 28), (24, 27, 28), (24, 25, 27), (25, 26, 27),
    # Nasal bridge to inner eyes & cheeks
    (18, 10, 11), (18, 11, 19), (19, 11, 12), (19, 12, 20), (20, 12, 13),
    (10, 22, 11), (22, 29, 11), (11, 29, 12), (29, 28, 12), (12, 28, 13),
    # Mid-face cheeks to upper lip
    (14, 21, 30), (21, 20, 30), (20, 31, 30), (20, 13, 31), (13, 32, 31), (13, 33, 32),
    (13, 34, 33), (13, 35, 34), (13, 28, 35), (28, 36, 35), (28, 27, 36), (27, 26, 36),
    # Outer lip to inner lip ring
    (30, 31, 42), (31, 43, 42), (31, 32, 43), (32, 33, 44), (32, 44, 43),
    (33, 34, 45), (33, 45, 44), (34, 35, 45), (35, 36, 46), (35, 46, 45),
    (36, 37, 46), (37, 47, 46), (37, 38, 47), (38, 39, 48), (38, 48, 47),
    (39, 40, 49), (39, 49, 48), (40, 41, 49), (41, 30, 42), (41, 42, 49),
    # Inner mouth cavity
    (42, 43, 49), (43, 48, 49), (43, 44, 48), (44, 45, 48), (45, 47, 48), (45, 46, 47),
)


@dataclass
class Face3DEvidence:
    vertices_3d: np.ndarray            # (50, 3) in upright pixel coordinates (u, v, z_px)
    normals_3d: np.ndarray             # (50, 3) rotated outward unit normals
    point_cos_view: np.ndarray         # (50,) dot(normal, +Z_camera)
    point_nose_ray_depth: np.ndarray   # (50,) normalized occlusion depth (z_nose - z_pt) / s >= 0
    point_nose_occluded: np.ndarray    # (50,) bool where normalized nose ray depth > 0.035
    point_ridge_crossed: np.ndarray    # (50,) bool where point crossed 3D nasal ridge crest
    far_side: str                      # "l" or "r"
    far_eye_cos_mean: float
    far_eye_cos_min: float
    far_brow_cos_mean: float
    far_brow_cos_min: float
    far_eye_nose_hits_3d: int
    far_eye_ridge_cross_3d: int
    far_corner_cos_mean: float
    vertices_norm: np.ndarray          # (50, 3) centered normalized coordinates for 3D viewer
    nasal_pyramid_norm: np.ndarray     # (12, 3) centered normalized nasal pyramid vertices

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertices_norm": np.round(self.vertices_norm, 4).tolist(),
            "normals": np.round(self.normals_3d, 4).tolist(),
            "point_cos_view": np.round(self.point_cos_view, 4).tolist(),
            "point_nose_ray_depth": np.round(self.point_nose_ray_depth, 4).tolist(),
            "point_nose_occluded": [bool(v) for v in self.point_nose_occluded],
            "point_ridge_crossed": [bool(v) for v in self.point_ridge_crossed],
            "nasal_pyramid_vertices": np.round(self.nasal_pyramid_norm, 4).tolist(),
            "nasal_pyramid_triangles": [list(t) for t in _NASAL_PYRAMID_TRIANGLES],
            "face_mesh_triangles": [list(t) for t in _FACE_MESH_TRIANGLES],
            "far_side": self.far_side,
            "far_eye_cos_mean": round(self.far_eye_cos_mean, 4),
            "far_eye_cos_min": round(self.far_eye_cos_min, 4),
            "far_brow_cos_mean": round(self.far_brow_cos_mean, 4),
            "far_brow_cos_min": round(self.far_brow_cos_min, 4),
            "far_eye_nose_hits_3d": int(self.far_eye_nose_hits_3d),
            "far_eye_ridge_cross_3d": int(self.far_eye_ridge_cross_3d),
            "far_corner_cos_mean": round(self.far_corner_cos_mean, 4),
        }


def _rotation_matrix_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Build 3D rotation matrix R = Rx(pitch) @ Ry(yaw) matching 6DRepNet/VF-50 conventions."""
    yr = math.radians(float(yaw_deg))
    pr = math.radians(float(pitch_deg))
    cy, sy = math.cos(yr), math.sin(yr)
    cp, sp = math.cos(pr), math.sin(pr)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return rx @ ry


def _raycast_triangle_z(
    px: float,
    py: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    eps: float = 0.02,
) -> float | None:
    """Return interpolated Z on triangle (a, b, c) at 2D pixel (px, py), or None if outside."""
    v0x, v0y = float(c[0] - a[0]), float(c[1] - a[1])
    v1x, v1y = float(b[0] - a[0]), float(b[1] - a[1])
    v2x, v2y = px - float(a[0]), py - float(a[1])

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-8:
        return None
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    if u >= -eps and v >= -eps and (u + v) <= (1.0 + eps):
        u_c = min(max(u, 0.0), 1.0)
        v_c = min(max(v, 0.0), 1.0 - u_c)
        return float(a[2] + u_c * (c[2] - a[2]) + v_c * (b[2] - a[2]))
    return None


def _build_nasal_pyramid_geometry(
    pts_3d: np.ndarray, eye_w_ref: float, scale_px: float
) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]]:
    """Construct 12 vertices & 12 triangles forming the personalized 3D left/right nasal pyramid walls."""
    ridge = pts_3d[10:14]
    z_inner_eye = min(float(pts_3d[18, 2]), float(pts_3d[22, 2]))
    z_base = min(z_inner_eye, float(ridge[0, 2]) - 0.16 * scale_px)

    base_l = np.zeros((4, 3), dtype=np.float64)
    base_r = np.zeros((4, 3), dtype=np.float64)
    for k in range(4):
        w_nasal = (0.14 + 0.04 * k) * eye_w_ref
        base_l[k] = [float(ridge[k, 0]) - w_nasal, float(ridge[k, 1]), z_base]
        base_r[k] = [float(ridge[k, 0]) + w_nasal, float(ridge[k, 1]), z_base]

    nasal_verts = np.vstack([ridge, base_l, base_r])
    tris = tuple(
        (nasal_verts[i], nasal_verts[j], nasal_verts[k])
        for i, j, k in _NASAL_PYRAMID_TRIANGLES
    )
    return nasal_verts, tris


class Face3DOccluder:
    """Zero-VRAM 2D-to-3D personalized facial mesh & nasal self-occlusion analyzer."""

    @staticmethod
    def analyze(local_pts_50: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0) -> Face3DEvidence:
        pts_xy = np.asarray(local_pts_50, dtype=np.float64)
        eye_l_w = max(float(np.linalg.norm(pts_xy[18] - pts_xy[14])), 1.0)
        eye_r_w = max(float(np.linalg.norm(pts_xy[26] - pts_xy[22])), 1.0)
        eye_w_ref = max(eye_l_w, eye_r_w)

        mouth_y = float(np.mean(pts_xy[30:42, 1]))
        eye_y = float(np.mean(pts_xy[14:30, 1]))
        scale_px = max((mouth_y - eye_y) / 0.62, eye_w_ref / 0.27, 16.0)

        rot = _rotation_matrix_yaw_pitch(yaw_deg, pitch_deg)
        v_rot = (_CANONICAL_VF50_XYZ - _HEAD_PIVOT) @ rot.T + _HEAD_PIVOT
        n_rot = _CANONICAL_VF50_NORMALS @ rot.T
        n_norm = np.linalg.norm(n_rot, axis=1, keepdims=True)
        n_rot = n_rot / np.maximum(n_norm, 1e-8)

        vertices_3d = np.column_stack([pts_xy[:, 0], pts_xy[:, 1], scale_px * v_rot[:, 2]])
        point_cos_view = n_rot[:, 2].astype(np.float64)

        nasal_verts_3d, tris = _build_nasal_pyramid_geometry(vertices_3d, eye_w_ref, scale_px)
        center_3d = np.mean(vertices_3d, axis=0)
        vertices_norm = (vertices_3d - center_3d) / scale_px
        nasal_pyramid_norm = (nasal_verts_3d - center_3d) / scale_px
        point_nose_ray_depth = np.zeros(50, dtype=np.float64)
        for pid in range(50):
            if 10 <= pid <= 13:
                continue
            px, py, pz = float(vertices_3d[pid, 0]), float(vertices_3d[pid, 1]), float(vertices_3d[pid, 2])
            best_z = -1e12
            for a, b, c in tris:
                z_hit = _raycast_triangle_z(px, py, a, b, c)
                if z_hit is not None and z_hit > best_z:
                    best_z = z_hit
            if best_z > pz:
                point_nose_ray_depth[pid] = (best_z - pz) / scale_px

        point_nose_occluded = point_nose_ray_depth > 0.035

        # Nasal ridge crest line sorted monotonically by vertical y for clean interpolation
        nose_y = pts_xy[10:14, 1]
        nose_x = pts_xy[10:14, 0]
        order = np.argsort(nose_y)
        nx_at_pts = np.interp(pts_xy[:, 1], nose_y[order], nose_x[order])

        far_side = "r" if yaw_deg >= 0.0 else "l"
        point_ridge_crossed = np.zeros(50, dtype=bool)
        if far_side == "r":
            far_brow_sl = slice(5, 10)
            far_eye_sl = slice(22, 30)
            far_corner_ids = (10, 35, 36, 37, 46)
            for pid in range(5, 10):
                point_ridge_crossed[pid] = bool(pts_xy[pid, 0] < nx_at_pts[pid] + 0.02 * eye_w_ref)
            for pid in range(22, 30):
                point_ridge_crossed[pid] = bool(pts_xy[pid, 0] < nx_at_pts[pid] + 0.02 * eye_w_ref)
        else:
            far_brow_sl = slice(0, 5)
            far_eye_sl = slice(14, 22)
            far_corner_ids = (10, 30, 31, 41, 42)
            for pid in range(0, 5):
                point_ridge_crossed[pid] = bool(pts_xy[pid, 0] > nx_at_pts[pid] - 0.02 * eye_w_ref)
            for pid in range(14, 22):
                point_ridge_crossed[pid] = bool(pts_xy[pid, 0] > nx_at_pts[pid] - 0.02 * eye_w_ref)

        return Face3DEvidence(
            vertices_3d=vertices_3d,
            normals_3d=n_rot,
            point_cos_view=point_cos_view,
            point_nose_ray_depth=point_nose_ray_depth,
            point_nose_occluded=point_nose_occluded,
            point_ridge_crossed=point_ridge_crossed,
            far_side=far_side,
            far_eye_cos_mean=float(np.mean(point_cos_view[far_eye_sl])),
            far_eye_cos_min=float(np.min(point_cos_view[far_eye_sl])),
            far_brow_cos_mean=float(np.mean(point_cos_view[far_brow_sl])),
            far_brow_cos_min=float(np.min(point_cos_view[far_brow_sl])),
            far_eye_nose_hits_3d=int(np.sum(point_nose_occluded[far_eye_sl])),
            far_eye_ridge_cross_3d=int(np.sum(point_ridge_crossed[far_eye_sl])),
            far_corner_cos_mean=float(np.mean(point_cos_view[list(far_corner_ids)])),
            vertices_norm=vertices_norm,
            nasal_pyramid_norm=nasal_pyramid_norm,
        )


def analyze_3d_occlusion(local_pts_50: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0) -> Face3DEvidence:
    """Convenience functional entry point for 2D-to-3D VF-50 self-occlusion analysis."""
    return Face3DOccluder.analyze(local_pts_50, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
