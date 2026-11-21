"""CVAT XML 1.1, JSON, and QC CSV exporter for VF-50 predictions."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from lxml import etree

from .vf50_rules import PointAnnotation, QCFlag
from .vf50_schema import SKELETON_SPECS, get_skeleton_edges


@dataclass
class ImagePrediction:
    image_name: str
    width: int
    height: int
    rotation_applied: str
    bbox_orig: Optional[List[float]]       # [x, y, w, h] in original image coordinates
    face_score: float
    used_clahe: bool
    multi_face: bool
    yaw: float
    pitch: float
    roll: float
    laplacian_var: float
    points: List[PointAnnotation]          # 50 points in original image coordinates
    qc_flags: List[QCFlag]
    detected: bool = True
    detector_used: str = "yunet"
    engine_used: str = "spiga_only"
    headpose_used: str = "spiga"
    mean_confidence: float = 1.0
    max_uncertainty_px: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.image_name,
            "width": self.width,
            "height": self.height,
            "rotation_applied": self.rotation_applied,
            "bbox_orig": self.bbox_orig,
            "face_score": round(self.face_score, 4),
            "used_clahe": self.used_clahe,
            "multi_face": self.multi_face,
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "laplacian_var": round(self.laplacian_var, 2),
            "detected": self.detected,
            "detector_used": self.detector_used,
            "engine_used": self.engine_used,
            "headpose_used": self.headpose_used,
            "mean_confidence": round(self.mean_confidence, 4),
            "max_uncertainty_px": round(self.max_uncertainty_px, 2),
            "points": [asdict(p) for p in self.points],
            "qc_flags": [asdict(f) for f in self.qc_flags],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImagePrediction":
        return cls(
            image_name=data["image_name"],
            width=int(data["width"]),
            height=int(data["height"]),
            rotation_applied=data.get("rotation_applied", "0"),
            bbox_orig=data.get("bbox_orig"),
            face_score=float(data.get("face_score", 0.0)),
            used_clahe=bool(data.get("used_clahe", False)),
            multi_face=bool(data.get("multi_face", False)),
            yaw=float(data.get("yaw", 0.0)),
            pitch=float(data.get("pitch", 0.0)),
            roll=float(data.get("roll", 0.0)),
            laplacian_var=float(data.get("laplacian_var", 0.0)),
            detected=bool(data.get("detected", True)),
            detector_used=data.get("detector_used", "yunet"),
            engine_used=data.get("engine_used", "spiga_only"),
            headpose_used=data.get("headpose_used", "spiga"),
            mean_confidence=float(data.get("mean_confidence", 1.0)),
            max_uncertainty_px=float(data.get("max_uncertainty_px", 0.0)),
            points=[PointAnnotation(**p) for p in data["points"]],
            qc_flags=[QCFlag(**f) for f in data.get("qc_flags", [])],
        )


def _build_skeleton_elements(pred: ImagePrediction) -> List[etree._Element]:
    """Create the 7 <skeleton> XML elements for one ImagePrediction."""
    pts_by_id: Dict[int, PointAnnotation] = {p.point_id: p for p in pred.points}
    elements: List[etree._Element] = []

    for spec in SKELETON_SPECS:
        skel_el = etree.Element(
            "skeleton",
            label=spec.name,
            source="auto",
            z_order="0",
        )
        for pid in spec.point_ids:
            pt = pts_by_id[pid]
            # Keep coordinates safely bounded inside image canvas for CVAT import stability
            cx = max(0.0, min(float(pred.width), float(pt.x)))
            cy = max(0.0, min(float(pred.height), float(pt.y)))
            out_val = 1 if int(pt.outside) else 0
            occ_val = 0 if out_val else (1 if int(pt.occluded) else 0)
            etree.SubElement(
                skel_el,
                "points",
                label=str(pid),
                source="auto",
                outside=str(out_val),
                occluded=str(occ_val),
                points=f"{cx:.2f},{cy:.2f}",
            )
        elements.append(skel_el)
    return elements


def _build_meta_element(num_images: int) -> etree._Element:
    """Build standard CVAT 1.1 <meta> block declaring all 7 VF-50 skeletons."""
    meta = etree.Element("meta")
    task = etree.SubElement(meta, "task")
    etree.SubElement(task, "name").text = "VF-50 Facial Landmarks"
    etree.SubElement(task, "size").text = str(num_images)
    etree.SubElement(task, "mode").text = "annotation"
    labels_el = etree.SubElement(task, "labels")

    for spec in SKELETON_SPECS:
        lbl = etree.SubElement(labels_el, "label")
        etree.SubElement(lbl, "name").text = spec.name
        etree.SubElement(lbl, "color").text = spec.stroke_hex
        etree.SubElement(lbl, "type").text = "skeleton"
        etree.SubElement(lbl, "attributes")

        # SVG skeleton template metadata
        svg_el = etree.SubElement(lbl, "svg")
        for idx, pid in enumerate(spec.point_ids):
            etree.SubElement(
                svg_el,
                "circle",
                id=str(pid),
                cx=f"{20 + idx * 15}",
                cy="30",
                r="1",
            )
        for pid_a, pid_b in get_skeleton_edges(spec):
            etree.SubElement(
                svg_el,
                "line",
                **{"data-node-from": str(pid_a), "data-node-to": str(pid_b)},
            )
    return meta


def export_cvat_xml(
    predictions: Iterable[ImagePrediction],
    output_xml_path: str,
    template_xml_path: Optional[str] = None,
) -> str:
    """Export predictions to CVAT for images 1.1 XML (optionally merging into a template XML)."""
    pred_list = list(predictions)
    os.makedirs(os.path.dirname(os.path.abspath(output_xml_path)), exist_ok=True)

    if template_xml_path:
        return export_with_template(template_xml_path, pred_list, output_xml_path)

    root = etree.Element("annotations")
    etree.SubElement(root, "version").text = "1.1"
    root.append(_build_meta_element(len(pred_list)))

    for idx, pred in enumerate(pred_list):
        img_el = etree.SubElement(
            root,
            "image",
            id=str(idx),
            name=pred.image_name,
            width=str(pred.width),
            height=str(pred.height),
        )
        if pred.detected and len(pred.points) == 50:
            for skel_el in _build_skeleton_elements(pred):
                img_el.append(skel_el)

    tree = etree.ElementTree(root)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
    return output_xml_path


def export_with_template(
    template_xml_path: str,
    predictions: Iterable[ImagePrediction],
    output_xml_path: str,
) -> str:
    """Inject predicted skeletons into an existing CVAT task annotations.xml matched by image name."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(template_xml_path, parser)
    root = tree.getroot()

    pred_by_name: Dict[str, ImagePrediction] = {}
    pred_by_basename: Dict[str, ImagePrediction] = {}
    for pred in predictions:
        pred_by_name[pred.image_name] = pred
        pred_by_basename[os.path.basename(pred.image_name)] = pred

    for img_el in root.findall("image"):
        name = img_el.get("name", "")
        pred = pred_by_name.get(name) or pred_by_basename.get(os.path.basename(name))
        if pred is None or not pred.detected or len(pred.points) != 50:
            continue

        # Remove old skeletons in this image before inserting updated pre-labels
        for old_skel in img_el.findall("skeleton"):
            img_el.remove(old_skel)
        for skel_el in _build_skeleton_elements(pred):
            img_el.append(skel_el)

    os.makedirs(os.path.dirname(os.path.abspath(output_xml_path)), exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
    return output_xml_path


def export_json(predictions: Iterable[ImagePrediction], output_json_path: str) -> str:
    """Save full prediction metadata and points to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    payload = [p.to_dict() for p in predictions]
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_json_path


def export_qc_csv(predictions: Iterable[ImagePrediction], output_csv_path: str) -> str:
    """Export QC summary CSV listing headpose, detection stats, and guideline issue flags."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_name",
                "detected",
                "rotation_applied",
                "face_score",
                "used_clahe",
                "yaw",
                "pitch",
                "roll",
                "laplacian_var",
                "visible_count",
                "occluded_count",
                "outside_count",
                "detector_used",
                "engine_used",
                "headpose_used",
                "mean_confidence",
                "max_uncertainty_px",
                "issue_types",
                "reasons",
            ]
        )
        for pred in predictions:
            vis_cnt = sum(1 for p in pred.points if p.outside == 0 and p.occluded == 0)
            occ_cnt = sum(1 for p in pred.points if p.outside == 0 and p.occluded == 1)
            out_cnt = sum(1 for p in pred.points if p.outside == 1)
            issue_types = ";".join(fl.issue_type for fl in pred.qc_flags)
            reasons = " | ".join(fl.reason for fl in pred.qc_flags)
            writer.writerow(
                [
                    pred.image_name,
                    int(pred.detected),
                    pred.rotation_applied,
                    f"{pred.face_score:.3f}",
                    int(pred.used_clahe),
                    f"{pred.yaw:.1f}",
                    f"{pred.pitch:.1f}",
                    f"{pred.roll:.1f}",
                    f"{pred.laplacian_var:.1f}",
                    vis_cnt,
                    occ_cnt,
                    out_cnt,
                    pred.detector_used,
                    pred.engine_used,
                    pred.headpose_used,
                    f"{pred.mean_confidence:.3f}",
                    f"{pred.max_uncertainty_px:.2f}",
                    issue_types,
                    reasons,
                ]
            )
    return output_csv_path
