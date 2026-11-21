"""Clean preview rendering for VF-50 predictions with distinct attribute markers."""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cvat_exporter import ImagePrediction
from .vf50_rules import PointAnnotation
from .vf50_schema import POINT_COLORS_HEX, SKELETON_SPECS, get_skeleton_edges

_COL_VISIBLE = (34, 197, 94)      # #22c55e
_COL_OCCLUDED = (245, 158, 11)    # #f59e0b
_COL_OCC_TEXT = (253, 224, 71)    # #fde047
_COL_OUTSIDE = (244, 63, 94)      # #f43f5e
_COL_OUT_TEXT = (253, 164, 175)   # #fda4af
_COL_UNC_MED = (232, 121, 249)    # #e879f9
_COL_UNC_HIGH = (255, 42, 133)    # #ff2a85


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    x, y = int(round(xy[0])), int(round(xy[1]))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    fill: Tuple[int, int, int],
    width: int = 2,
    dash: int = 6,
    gap: int = 4,
) -> None:
    """Draw a dashed or dotted segment between two endpoints."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    step = float(dash + gap)
    while pos < dist:
        end = min(pos + dash, dist)
        sx, sy = x1 + ux * pos, y1 + uy * pos
        ex, ey = x1 + ux * end, y1 + uy * end
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += step


def _draw_diamond(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    r: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int = 2,
) -> None:
    """Draw a diamond marker (Occluded) with a white center dot."""
    d = r * 1.4
    poly = [(x, y - d), (x + d, y), (x, y + d), (x - d, y)]
    draw.polygon(poly, fill=fill, outline=outline, width=width)
    draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 255, 255))


def _draw_outside_marker(draw: ImageDraw.ImageDraw, x: float, y: float, r: float) -> None:
    """Draw an Outside marker: rose circle outline + X cross."""
    rr = r + 1.0
    draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(76, 5, 25), outline=_COL_OUTSIDE, width=2)
    cr = max(3.0, rr * 0.55)
    draw.line([(x - cr, y - cr), (x + cr, y + cr)], fill=_COL_OUT_TEXT, width=2)
    draw.line([(x + cr, y - cr), (x - cr, y + cr)], fill=_COL_OUT_TEXT, width=2)


def _draw_bottom_legend(draw: ImageDraw.ImageDraw, width: int, height: int, font: ImageFont.ImageFont) -> None:
    """Render compact attribute legend bar along the bottom edge of the preview."""
    bar_h = 22
    y0 = max(24, height - bar_h)
    cy = y0 + bar_h * 0.5
    draw.rectangle([0, y0, width, height], fill=(16, 20, 26))

    x = 10.0
    # 1. Visible
    draw.ellipse([x - 5, cy - 5, x + 5, cy + 5], fill=(0, 0, 0))
    draw.ellipse([x - 3.8, cy - 3.8, x + 3.8, cy + 3.8], fill=_COL_VISIBLE)
    _draw_text(draw, (x + 8, y0 + 4), "Visible", font, (134, 239, 172))
    x += 64

    # 2. Occluded
    if x + 115 < width:
        _draw_diamond(draw, x, cy, 4.0, _COL_OCCLUDED, (255, 255, 255), width=1)
        _draw_text(draw, (x + 8, y0 + 4), "Occluded (nét đứt)", font, _COL_OCC_TEXT)
        x += 126

    # 3. Outside
    if x + 78 < width:
        _draw_outside_marker(draw, x, cy, 3.8)
        _draw_text(draw, (x + 8, y0 + 4), "Outside", font, _COL_OUT_TEXT)
        x += 72

    # 4. AI Uncertainty
    if x + 95 < width:
        draw.ellipse([x - 5, cy - 5, x + 5, cy + 5], outline=_COL_UNC_MED, width=2)
        draw.ellipse([x - 2, cy - 2, x + 2, cy + 2], fill=_COL_UNC_HIGH)
        _draw_text(draw, (x + 8, y0 + 4), "AI Conf<65%", font, (245, 208, 254))


def render_preview(
    bgr_image: np.ndarray,
    pred: ImagePrediction,
    show_outside: bool = False,
    color_mode: str = "state",
) -> Image.Image:
    """Render annotated preview image (RGB PIL Image) with clear attribute shapes and legend."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font_id = _load_font(11)
    font_hdr = _load_font(13)
    font_leg = _load_font(11)

    if pred.bbox_orig:
        bx, by, bw, bh = pred.bbox_orig
        draw.rectangle([bx, by, bx + bw, by + bh], outline=(70, 160, 255), width=1)

    pts_map: Dict[int, PointAnnotation] = {p.point_id: p for p in pred.points}
    skel_colors: Dict[str, Tuple[int, int, int]] = {
        spec.name: _hex_to_rgb(spec.stroke_hex) for spec in SKELETON_SPECS
    }

    # Draw skeleton edges: Visible solid / Occluded dashed / Outside dotted
    for spec in SKELETON_SPECS:
        edge_rgb = _hex_to_rgb(spec.stroke_hex)
        for pid_a, pid_b in get_skeleton_edges(spec):
            pa, pb = pts_map.get(pid_a), pts_map.get(pid_b)
            if not pa or not pb:
                continue
            has_out = pa.outside == 1 or pb.outside == 1
            if has_out and not show_outside:
                continue
            has_occ = (not has_out) and (pa.occluded == 1 or pb.occluded == 1)
            if has_out:
                _draw_dashed_line(draw, (pa.x, pa.y), (pb.x, pb.y), _COL_OUTSIDE, width=1, dash=2, gap=4)
            elif has_occ:
                occ_edge_rgb = _COL_OCCLUDED if color_mode == "state" else edge_rgb
                _draw_dashed_line(draw, (pa.x, pa.y), (pb.x, pb.y), occ_edge_rgb, width=2, dash=6, gap=4)
            else:
                draw.line([(pa.x, pa.y), (pb.x, pb.y)], fill=edge_rgb, width=2)

    # Draw points with state-distinct geometry and uncertainty rings
    r = 4.5
    for pid in range(50):
        pt = pts_map.get(pid)
        if not pt:
            continue
        if pt.outside == 1 and not show_outside:
            continue

        if pt.outside == 0 and pt.confidence < 0.65:
            if pt.confidence < 0.45:
                hr1, hr2 = r + 4.5, r + 7.5
                draw.ellipse([pt.x - hr1, pt.y - hr1, pt.x + hr1, pt.y + hr1], outline=_COL_UNC_HIGH, width=2)
                draw.ellipse([pt.x - hr2, pt.y - hr2, pt.x + hr2, pt.y + hr2], outline=_COL_UNC_HIGH, width=1)
                _draw_text(draw, (pt.x - 2, pt.y - hr2 - 11), "!", font_id, _COL_UNC_HIGH)
            else:
                hr = r + 4.5
                draw.ellipse([pt.x - hr, pt.y - hr, pt.x + hr, pt.y + hr], outline=_COL_UNC_MED, width=2)

        if color_mode == "skeleton":
            core_rgb = skel_colors.get(pt.skeleton, _COL_VISIBLE)
        elif color_mode == "cvat":
            core_rgb = _hex_to_rgb(POINT_COLORS_HEX[pid])
        else:
            core_rgb = _COL_OCCLUDED if pt.occluded == 1 else _COL_VISIBLE

        if pt.outside == 1:
            _draw_outside_marker(draw, pt.x, pt.y, r)
        elif pt.occluded == 1:
            _draw_diamond(draw, pt.x, pt.y, r, core_rgb, (255, 255, 255), width=2)
        else:
            draw.ellipse([pt.x - r - 1.5, pt.y - r - 1.5, pt.x + r + 1.5, pt.y + r + 1.5], fill=(0, 0, 0))
            draw.ellipse([pt.x - r, pt.y - r, pt.x + r, pt.y + r], fill=core_rgb)

    # Draw non-overlapping ID hints with O / x state suffixes
    mouth_w = abs(pts_map[36].x - pts_map[30].x) if 30 in pts_map and 36 in pts_map else 0.0
    for pid in range(50):
        pt = pts_map.get(pid)
        if not pt:
            continue
        if pt.outside == 1 and not show_outside:
            continue
        if 42 <= pid <= 49 and mouth_w < 95.0 and pid not in (42, 44, 46, 48):
            continue
        if pt.outside == 1:
            txt_col, label_txt = _COL_OUT_TEXT, f"{pid}x"
        elif pt.occluded == 1:
            txt_col, label_txt = _COL_OCC_TEXT, f"{pid}O"
        else:
            txt_col, label_txt = (255, 255, 255), str(pid)
        offset_y = 8 if pid in (19, 20, 21, 27, 28, 29, 37, 38, 39, 40, 41, 47, 48, 49) else -14
        _draw_text(draw, (pt.x + 6, pt.y + offset_y), label_txt, font_id, txt_col)

    # Top header bar & bottom legend bar
    vis_cnt = sum(1 for p in pred.points if p.outside == 0 and p.occluded == 0)
    occ_cnt = sum(1 for p in pred.points if p.outside == 0 and p.occluded == 1)
    out_cnt = sum(1 for p in pred.points if p.outside == 1)
    flags_str = ", ".join(f.issue_type for f in pred.qc_flags) if pred.qc_flags else "OK"
    conf_str = f" | Conf:{int(round(pred.mean_confidence * 100))}%" if pred.detected else ""
    hdr = (
        f"{pred.image_name} | rot:{pred.rotation_applied} | "
        f"yaw:{pred.yaw:.1f} pitch:{pred.pitch:.1f} roll:{pred.roll:.1f} | "
        f"V:{vis_cnt} O:{occ_cnt} Out:{out_cnt}{conf_str} | {flags_str}"
    )
    draw.rectangle([0, 0, img.width, 24], fill=(16, 20, 26))
    _draw_text(draw, (8, 4), hdr, font_hdr, (140, 255, 170) if pred.detected else (255, 120, 120))
    _draw_bottom_legend(draw, img.width, img.height, font_leg)
    return img
