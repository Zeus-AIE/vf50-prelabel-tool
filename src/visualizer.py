"""Clean preview rendering for VF-50 predictions."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cvat_exporter import ImagePrediction
from .vf50_rules import PointAnnotation
from .vf50_schema import POINT_COLORS_HEX, SKELETON_SPECS, get_skeleton_edges


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


def render_preview(bgr_image: np.ndarray, pred: ImagePrediction) -> Image.Image:
    """Render annotated preview image (RGB PIL Image) on original image canvas."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font_id = _load_font(11)
    font_hdr = _load_font(13)

    if pred.bbox_orig:
        bx, by, bw, bh = pred.bbox_orig
        draw.rectangle([bx, by, bx + bw, by + bh], outline=(70, 160, 255), width=1)

    pts_map: Dict[int, PointAnnotation] = {p.point_id: p for p in pred.points}

    # Draw skeleton edges where both endpoints are not Outside
    for spec in SKELETON_SPECS:
        edge_rgb = _hex_to_rgb(spec.stroke_hex)
        for pid_a, pid_b in get_skeleton_edges(spec):
            pa, pb = pts_map.get(pid_a), pts_map.get(pid_b)
            if pa and pb and pa.outside == 0 and pb.outside == 0:
                draw.line([(pa.x, pa.y), (pb.x, pb.y)], fill=edge_rgb, width=2)

    # Draw points (Visible = black border, Occluded = white border, Outside = hidden)
    r = 4
    for pid in range(50):
        pt = pts_map.get(pid)
        if not pt or pt.outside == 1:
            continue
        if pt.confidence < 0.65:
            halo_rgb = (249, 115, 22) if pt.confidence >= 0.45 else (239, 68, 68)
            hr = r + 4
            draw.ellipse([pt.x - hr, pt.y - hr, pt.x + hr, pt.y + hr], outline=halo_rgb, width=2)
        fill_rgb = _hex_to_rgb(POINT_COLORS_HEX[pid])
        border_rgb = (255, 255, 255) if pt.occluded == 1 else (0, 0, 0)
        draw.ellipse([pt.x - r - 1, pt.y - r - 1, pt.x + r + 1, pt.y + r + 1], fill=border_rgb)
        draw.ellipse([pt.x - r, pt.y - r, pt.x + r, pt.y + r], fill=fill_rgb)

    # Draw non-overlapping ID hints (skip crowded inner lip IDs if mouth < 60px wide unless zoomed, or place cleanly)
    mouth_w = abs(pts_map[36].x - pts_map[30].x) if 30 in pts_map and 36 in pts_map else 0.0
    for pid in range(50):
        pt = pts_map.get(pid)
        if not pt or pt.outside == 1:
            continue
        if 42 <= pid <= 49 and mouth_w < 95.0 and pid not in (42, 44, 46, 48):
            continue
        txt_col = (140, 235, 255) if pt.occluded == 1 else (255, 255, 255)
        offset_y = 8 if pid in (19, 20, 21, 27, 28, 29, 37, 38, 39, 40, 41, 47, 48, 49) else -14
        _draw_text(draw, (pt.x + 5, pt.y + offset_y), str(pid), font_id, txt_col)

    # Header status bar
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
    return img
