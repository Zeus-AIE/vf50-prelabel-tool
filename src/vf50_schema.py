"""Schema definition and WFLW-98 to VF-50 mapping for CVAT skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    point_ids: Tuple[int, ...]
    closed: bool
    min_visible_threshold: int  # Table 3.1 minimum visible points threshold
    stroke_hex: str


SKELETON_SPECS: Tuple[SkeletonSpec, ...] = (
    SkeletonSpec("longmaytrai", (0, 1, 2, 3, 4), False, 3, "#7a64fd"),
    SkeletonSpec("longmayphai", (5, 6, 7, 8, 9), False, 3, "#b8f144"),
    SkeletonSpec("songmui", (10, 11, 12, 13), False, 2, "#5cc84f"),
    SkeletonSpec("mattrai", (14, 15, 16, 17, 18, 19, 20, 21), True, 4, "#c99546"),
    SkeletonSpec("matphai", (22, 23, 24, 25, 26, 27, 28, 29), True, 4, "#68e631"),
    SkeletonSpec("moingoai", (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), True, 6, "#2b6bb4"),
    SkeletonSpec("moitrong", (42, 43, 44, 45, 46, 47, 48, 49), True, 4, "#367a82"),
)

SKELETON_BY_NAME: Dict[str, SkeletonSpec] = {s.name: s for s in SKELETON_SPECS}

# Exact 1-to-1 mapping from WFLW-98 index -> VF-50 Point ID (0..49), verified against SPIGA db_info.json
VF50_TO_WFLW98: Tuple[int, ...] = (
    33, 34, 35, 36, 37,                          # 0-4:   longmaytrai (outer -> inner)
    42, 43, 44, 45, 46,                          # 5-9:   longmayphai (inner -> outer)
    51, 52, 53, 54,                              # 10-13: songmui (top -> base)
    60, 61, 62, 63, 64, 65, 66, 67,              # 14-21: mattrai (outer corner, clockwise)
    68, 69, 70, 71, 72, 73, 74, 75,              # 22-29: matphai (inner corner, clockwise)
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,  # 30-41: moingoai (left corner, clockwise)
    88, 89, 90, 91, 92, 93, 94, 95,              # 42-49: moitrong (left corner, clockwise)
)

# Distinct node colors (matching CVAT display conventions)
POINT_COLORS_HEX: Tuple[str, ...] = (
    "#7d050a", "#d12345", "#350dea", "#479ffe", "#4a649f",
    "#478144", "#57236b", "#1cdda5", "#e2bc6e", "#f067db",
    "#63bbfa", "#22b16f", "#daddec", "#2ac791",
    "#de22a0", "#a7a570", "#74db1b", "#aaa246", "#bfa552", "#9b8d37", "#c8023c", "#c0762a",
    "#76bb18", "#afc7d8", "#3b1ebf", "#3d4e6a", "#b295e8", "#a8eeed", "#1cb413", "#b19fbd",
    "#5cc934", "#179232", "#5d93c8", "#0b9b66", "#1004d4", "#76ffa6", "#e52057", "#0ca4a0", "#34fb06", "#ff6165", "#049dea", "#7dd43b",
    "#a62183", "#fbcfc4", "#7b1509", "#ed30ca", "#a73a69", "#9553cb", "#623d5e", "#973ab6",
)


def get_skeleton_edges(spec: SkeletonSpec) -> List[Tuple[int, int]]:
    """Return list of (point_id_a, point_id_b) edges for a skeleton."""
    ids = spec.point_ids
    edges = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    if spec.closed and len(ids) > 2:
        edges.append((ids[-1], ids[0]))
    return edges


def point_id_to_skeleton(point_id: int) -> str:
    """Return the parent skeleton name for a given Point ID (0..49)."""
    for spec in SKELETON_SPECS:
        if point_id in spec.point_ids:
            return spec.name
    raise ValueError(f"Unknown VF-50 point_id: {point_id}")


# WFLW-98 horizontal flip permutation order (verified against SPIGA db_info.json)
WFLW98_FLIP_ORDER: Tuple[int, ...] = (
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    46, 45, 44, 43, 42, 50, 49, 48, 47,
    37, 36, 35, 34, 33, 41, 40, 39, 38,
    51, 52, 53, 54,
    59, 58, 57, 56, 55,
    72, 71, 70, 69, 68, 75, 74, 73,
    64, 63, 62, 61, 60, 67, 66, 65,
    82, 81, 80, 79, 78, 77, 76,
    87, 86, 85, 84, 83,
    92, 91, 90, 89, 88,
    95, 94, 93,
    97, 96,
)
_WFLW98_FLIP_INDEX = np.array(WFLW98_FLIP_ORDER, dtype=np.int64)


def flip_wflw98_points(pts_98: np.ndarray, width: float, align_corners: bool = False) -> np.ndarray:
    """Un-flip WFLW-98 landmarks predicted on a horizontally flipped image of given width."""
    out = np.array(pts_98, dtype=np.float64)[_WFLW98_FLIP_INDEX].copy()
    max_x = (width - 1.0) if align_corners else width
    out[:, 0] = max_x - out[:, 0]
    return out
