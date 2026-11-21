"""FastAPI local Web UI backend for interactive VF-50 pre-labeling and CVAT export."""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .cvat_exporter import (
    ImagePrediction,
    export_cvat_xml,
    export_json,
    export_qc_csv,
)
from .detector import VALID_DETECTORS, VALID_DRIVER_SIDES, VALID_ROTATIONS
from .pipeline import (
    IMAGE_EXTS,
    VALID_ENSEMBLES,
    VALID_HEADPOSES,
    VF50Pipeline,
    collect_images,
    get_level2_capabilities,
    get_level_presets,
    imread_unicode,
    list_available_weights,
    resolve_default_weights,
)
from .vf50_rules import PointAnnotation, revalidate_prediction_qc
from .vf50_schema import POINT_COLORS_HEX, SKELETON_SPECS, get_skeleton_edges

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("VF50_MODELS_DIR", os.path.join(ROOT_DIR, "models"))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
CACHE_DIR = os.path.join(ROOT_DIR, ".ui_cache")
UPLOAD_DIR = os.path.join(CACHE_DIR, "uploads")
PASTE_DIR = os.path.join(CACHE_DIR, "pastes")
EXPORT_DIR = os.path.join(CACHE_DIR, "exports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PASTE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


@dataclass
class SessionImage:
    image_id: int
    image_name: str
    file_path: str
    width: int
    height: int
    prediction: Optional[ImagePrediction] = None


class SessionState:
    def __init__(self) -> None:
        self.images: List[SessionImage] = []
        self.source_dir: Optional[str] = None
        self.pipeline: Optional[VF50Pipeline] = None
        self.current_weights: Optional[str] = None

    def get_pipeline(
        self, device: str = "cuda", weights_name: Optional[str] = None
    ) -> VF50Pipeline:
        available = list_available_weights(MODELS_DIR)
        target = weights_name or resolve_default_weights(MODELS_DIR)
        if target not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Weights '{target}' không hợp lệ hoặc không tồn tại trong {MODELS_DIR}",
            )

        if self.pipeline is None:
            self.pipeline = VF50Pipeline(
                models_dir=MODELS_DIR, device=device, weights_name=target
            )
        elif self.current_weights != target:
            self.pipeline.reload_spiga_weights(target)
        self.current_weights = self.pipeline.weights_name
        return self.pipeline

    def get_image(self, image_id: int) -> SessionImage:
        for item in self.images:
            if item.image_id == image_id:
                return item
        raise HTTPException(status_code=404, detail=f"Image id {image_id} not found")


SESSION = SessionState()
app = FastAPI(title="VF-50 Local Pre-label Tool", version="1.0.0")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LoadFolderRequest(BaseModel):
    folder_path: str


class PredictRequest(BaseModel):
    image_id: int
    rotate_mode: str = "0"
    driver_side: str = "largest"
    yaw_occluded: float = 35.0
    yaw_outside: float = 55.0
    manual_bbox: Optional[List[float]] = None
    weights_name: Optional[str] = None
    detector_mode: Optional[str] = None
    ensemble_mode: Optional[str] = None
    headpose_mode: Optional[str] = None
    use_tta: Optional[bool] = None
    use_two_pass: Optional[bool] = None


class PredictBatchRequest(BaseModel):
    rotate_mode: str = "0"
    driver_side: str = "largest"
    yaw_occluded: float = 35.0
    yaw_outside: float = 55.0
    weights_name: Optional[str] = None
    detector_mode: Optional[str] = None
    ensemble_mode: Optional[str] = None
    headpose_mode: Optional[str] = None
    use_tta: Optional[bool] = None
    use_two_pass: Optional[bool] = None


class PointUpdateItem(BaseModel):
    point_id: int
    x: float
    y: float
    occluded: int
    outside: int


class UpdateImageRequest(BaseModel):
    points: List[PointUpdateItem]
    bbox_orig: Optional[List[float]] = None


def _serialize_session_item(item: SessionImage) -> Dict[str, Any]:
    return {
        "image_id": item.image_id,
        "image_name": item.image_name,
        "width": item.width,
        "height": item.height,
        "has_prediction": item.prediction is not None,
        "prediction": item.prediction.to_dict() if item.prediction else None,
    }


@app.get("/")
def index() -> FileResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="static/index.html missing")
    return FileResponse(index_path)


@app.get("/api/schema")
def get_schema() -> Dict[str, Any]:
    caps = get_level2_capabilities(MODELS_DIR)
    presets = get_level_presets(MODELS_DIR)
    return {
        "skeletons": [
            {
                "name": s.name,
                "point_ids": list(s.point_ids),
                "closed": s.closed,
                "min_visible_threshold": s.min_visible_threshold,
                "stroke_hex": s.stroke_hex,
                "edges": [list(e) for e in get_skeleton_edges(s)],
            }
            for s in SKELETON_SPECS
        ],
        "point_colors": list(POINT_COLORS_HEX),
        "rotations": list(VALID_ROTATIONS),
        "driver_sides": list(VALID_DRIVER_SIDES),
        "available_weights": list_available_weights(MODELS_DIR),
        "default_weights": resolve_default_weights(MODELS_DIR),
        **caps,
        **presets,
    }


@app.get("/api/images")
def list_images() -> Dict[str, Any]:
    return {
        "source_dir": SESSION.source_dir,
        "images": [_serialize_session_item(item) for item in SESSION.images],
    }


@app.get("/api/image/{image_id}")
def get_image_file(image_id: int) -> FileResponse:
    item = SESSION.get_image(image_id)
    if not os.path.exists(item.file_path):
        raise HTTPException(status_code=404, detail="Image file no longer exists on disk")
    return FileResponse(item.file_path)


@app.post("/api/load-folder")
def load_folder(req: LoadFolderRequest) -> Dict[str, Any]:
    folder = req.folder_path.strip().strip('"')
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Thư mục không tồn tại: {folder}")

    paths = collect_images(folder)
    if not paths:
        raise HTTPException(status_code=400, detail=f"Không tìm thấy ảnh hợp lệ trong: {folder}")

    new_items: List[SessionImage] = []
    for idx, p in enumerate(paths):
        img = imread_unicode(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        new_items.append(
            SessionImage(
                image_id=idx,
                image_name=p.name,
                file_path=str(p),
                width=w,
                height=h,
            )
        )
    if not new_items:
        raise HTTPException(status_code=400, detail="Không đọc được file ảnh nào trong thư mục")

    SESSION.images = new_items
    SESSION.source_dir = folder
    return {
        "count": len(SESSION.images),
        "source_dir": SESSION.source_dir,
        "images": [_serialize_session_item(item) for item in SESSION.images],
    }


@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    # Clean upload staging directory for fresh upload batch
    for old in Path(UPLOAD_DIR).glob("*"):
        if old.is_file():
            old.unlink()

    new_items: List[SessionImage] = []
    idx = 0
    for up in files:
        filename = os.path.basename(up.filename or f"img_{idx}.png")
        ext = Path(filename).suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        dst = os.path.join(UPLOAD_DIR, filename)
        with open(dst, "wb") as out_f:
            shutil.copyfileobj(up.file, out_f)
        img = imread_unicode(dst)
        if img is None:
            continue
        h, w = img.shape[:2]
        new_items.append(
            SessionImage(
                image_id=idx,
                image_name=filename,
                file_path=dst,
                width=w,
                height=h,
            )
        )
        idx += 1

    if not new_items:
        raise HTTPException(status_code=400, detail="Không có file ảnh hợp lệ được tải lên")

    SESSION.images = new_items
    SESSION.source_dir = UPLOAD_DIR
    return {
        "count": len(SESSION.images),
        "images": [_serialize_session_item(item) for item in SESSION.images],
    }


@app.post("/api/paste-image")
async def paste_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Append a clipboard screenshot or dropped image without clearing existing session images."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Không nhận được dữ liệu ảnh từ clipboard")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Dữ liệu dán không phải ảnh hợp lệ")

    orig_ext = Path(os.path.basename(file.filename or "")).suffix.lower()
    ext = orig_ext if orig_ext in IMAGE_EXTS else ".png"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"paste_{ts}{ext}"
    dst = os.path.join(PASTE_DIR, fname)
    suffix_i = 1
    while os.path.exists(dst):
        fname = f"paste_{ts}_{suffix_i}{ext}"
        dst = os.path.join(PASTE_DIR, fname)
        suffix_i += 1

    with open(dst, "wb") as out_f:
        out_f.write(raw)

    h, w = img.shape[:2]
    next_id = (max(it.image_id for it in SESSION.images) + 1) if SESSION.images else 0
    new_item = SessionImage(
        image_id=next_id,
        image_name=fname,
        file_path=dst,
        width=w,
        height=h,
    )
    SESSION.images.append(new_item)
    return {
        "image": _serialize_session_item(new_item),
        "index": len(SESSION.images) - 1,
        "count": len(SESSION.images),
        "images": [_serialize_session_item(it) for it in SESSION.images],
    }


@app.post("/api/clear-images")
def clear_images() -> Dict[str, Any]:
    """Clear current image list from session."""
    SESSION.images = []
    SESSION.source_dir = None
    return {"count": 0, "images": []}


def _validate_mode_options(
    detector_mode: Optional[str],
    ensemble_mode: Optional[str],
    headpose_mode: Optional[str],
) -> None:
    if detector_mode is not None and detector_mode not in VALID_DETECTORS:
        raise HTTPException(status_code=400, detail=f"detector_mode không hợp lệ: {detector_mode}")
    if ensemble_mode is not None and ensemble_mode not in VALID_ENSEMBLES:
        raise HTTPException(status_code=400, detail=f"ensemble_mode không hợp lệ: {ensemble_mode}")
    if headpose_mode is not None and headpose_mode not in VALID_HEADPOSES:
        raise HTTPException(status_code=400, detail=f"headpose_mode không hợp lệ: {headpose_mode}")


@app.post("/api/predict")
def predict_single(req: PredictRequest) -> Dict[str, Any]:
    _validate_mode_options(req.detector_mode, req.ensemble_mode, req.headpose_mode)
    item = SESSION.get_image(req.image_id)
    pipeline = SESSION.get_pipeline(weights_name=req.weights_name)
    pred = pipeline.predict_image(
        image_path=item.file_path,
        image_name=item.image_name,
        rotate_mode=req.rotate_mode,
        driver_side=req.driver_side,
        yaw_occluded=req.yaw_occluded,
        yaw_outside=req.yaw_outside,
        manual_bbox=req.manual_bbox,
        detector_mode=req.detector_mode,
        ensemble_mode=req.ensemble_mode,
        headpose_mode=req.headpose_mode,
        use_tta=req.use_tta,
        use_two_pass=req.use_two_pass,
    )
    item.prediction = pred
    return _serialize_session_item(item)


@app.post("/api/predict-batch")
def predict_batch(req: PredictBatchRequest) -> Dict[str, Any]:
    _validate_mode_options(req.detector_mode, req.ensemble_mode, req.headpose_mode)
    if not SESSION.images:
        raise HTTPException(status_code=400, detail="Chưa có ảnh nào được nạp vào phiên làm việc")
    pipeline = SESSION.get_pipeline(weights_name=req.weights_name)
    for item in SESSION.images:
        item.prediction = pipeline.predict_image(
            image_path=item.file_path,
            image_name=item.image_name,
            rotate_mode=req.rotate_mode,
            driver_side=req.driver_side,
            yaw_occluded=req.yaw_occluded,
            yaw_outside=req.yaw_outside,
            detector_mode=req.detector_mode,
            ensemble_mode=req.ensemble_mode,
            headpose_mode=req.headpose_mode,
            use_tta=req.use_tta,
            use_two_pass=req.use_two_pass,
        )
    detected = sum(1 for it in SESSION.images if it.prediction and it.prediction.detected)
    return {
        "total": len(SESSION.images),
        "detected": detected,
        "images": [_serialize_session_item(it) for it in SESSION.images],
    }


@app.post("/api/update/{image_id}")
def update_image_annotation(image_id: int, req: UpdateImageRequest) -> Dict[str, Any]:
    item = SESSION.get_image(image_id)
    if item.prediction is None:
        raise HTTPException(status_code=400, detail="Ảnh này chưa có dữ liệu pre-label để cập nhật")

    pts_map: Dict[int, PointAnnotation] = {p.point_id: p for p in item.prediction.points}
    for upd in req.points:
        if upd.point_id in pts_map:
            pt = pts_map[upd.point_id]
            pt.x = float(upd.x)
            pt.y = float(upd.y)
            pt.occluded = 1 if upd.occluded else 0
            pt.outside = 1 if upd.outside else 0
            if pt.outside == 1:
                pt.occluded = 0
    if req.bbox_orig is not None:
        item.prediction.bbox_orig = [float(v) for v in req.bbox_orig]
        item.prediction.detected = True

    active_pts = [p for p in item.prediction.points if p.outside == 0] or item.prediction.points
    item.prediction.mean_confidence = float(np.mean([p.confidence for p in active_pts]))
    item.prediction.max_uncertainty_px = float(np.max([p.uncertainty_px for p in active_pts]))
    item.prediction.qc_flags = revalidate_prediction_qc(
        item.prediction.points,
        item.prediction.width,
        item.prediction.height,
        item.prediction.qc_flags,
    )

    return _serialize_session_item(item)


@app.post("/api/export-cvat")
async def export_cvat(
    output_dir: Optional[str] = Form(default=None),
    template_xml: Optional[UploadFile] = File(default=None),
) -> Response:
    preds = [it.prediction for it in SESSION.images if it.prediction is not None]
    if not preds:
        raise HTTPException(status_code=400, detail="Chưa có ảnh nào được pre-label để xuất")

    target_dir = output_dir.strip().strip('"') if output_dir and output_dir.strip() else EXPORT_DIR
    os.makedirs(target_dir, exist_ok=True)
    xml_out = os.path.join(target_dir, "annotations.xml")

    tmp_tpl_path: Optional[str] = None
    try:
        if template_xml is not None and template_xml.filename:
            fd, tmp_tpl_path = tempfile.mkstemp(suffix=".xml")
            os.close(fd)
            with open(tmp_tpl_path, "wb") as tf:
                shutil.copyfileobj(template_xml.file, tf)

        export_cvat_xml(preds, xml_out, template_xml_path=tmp_tpl_path)
        export_json(preds, os.path.join(target_dir, "predictions.json"))
        export_qc_csv(preds, os.path.join(target_dir, "qc_report.csv"))

        with open(xml_out, "rb") as f:
            xml_bytes = f.read()
        return Response(
            content=xml_bytes,
            media_type="application/xml",
            headers={
                "Content-Disposition": 'attachment; filename="annotations.xml"',
                "X-Export-Dir": target_dir,
            },
        )
    finally:
        if tmp_tpl_path and os.path.exists(tmp_tpl_path):
            os.remove(tmp_tpl_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.web_app:app", host="127.0.0.1", port=7860, reload=False)
