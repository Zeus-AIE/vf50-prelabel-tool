"""Unified Model Downloader for VF-50 Pre-labeling Tool.

Downloads and verifies all pretrained weights for Level 1, Level 2, and Level 3:
1. YuNet (face_detection_yunet_2023mar.onnx) - Fast face detection
2. SCRFD-10G (scrfd_10g_bnkps.onnx) - Robust IR face detection
3. SPIGA Base (spiga_wflw.pt) - Base 98-point facial landmark model
4. 6DRepNet (6DRepNet_300W_LP_AFLW2000.pth) - Continuous 3D headpose estimation
5. TUFA (tufa_face_prompt_base.pth & shape_98.npz) - Transformer-based landmark refiner
6. SPIGA-IR Fine-tuned (spiga_wflw_ir_vf50.pt) - Level 3 IR-optimized weights
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import gdown
import numpy as np
import onnxruntime as ort
import requests
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def _download_http(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        ct = r.headers.get("Content-Type", "")
        if "text/html" in ct.lower() and dst.suffix in {".pt", ".pth", ".onnx", ".npz"}:
            raise RuntimeError(f"Unexpected HTML response from {url} (Content-Type={ct})")
        with open(tmp, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=dst.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, dst)


def download_with_mirrors(urls: Iterable[str], dst: Path, min_bytes: int, optional: bool = False) -> bool:
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"  [OK] {dst.name} is already present ({dst.stat().st_size / (1024**2):.2f} MB)")
        return True

    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"  [GET] Downloading {dst.name} from {url} ...")
            _download_http(url, dst)
            size = dst.stat().st_size
            if size < min_bytes:
                raise RuntimeError(f"{dst.name} size too small: {size} < {min_bytes}")
            print(f"  [SUCCESS] {dst.name} ({size / (1024**2):.2f} MB)")
            return True
        except Exception as e:
            last_err = e
            print(f"  [WARN] Failed {url}: {e}")

    if optional:
        print(f"  [INFO] Optional model {dst.name} not downloaded yet: {last_err}")
        return False
    raise RuntimeError(f"Failed downloading required model {dst.name}: {last_err}")


def download_gdrive(file_id: str, dst: Path, min_bytes: int) -> Path:
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"  [OK] {dst.name} is already present ({dst.stat().st_size / (1024**2):.2f} MB)")
        return dst

    direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        if download_with_mirrors([direct_url], dst, min_bytes=min_bytes, optional=True):
            return dst
    except Exception:
        pass

    print(f"  [GET] Fetching {dst.name} via gdown (Google Drive ID: {file_id}) ...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dst), quiet=False)
    if not dst.exists() or dst.stat().st_size < min_bytes:
        raise RuntimeError(f"Failed to download {dst.name} from Google Drive")
    print(f"  [SUCCESS] {dst.name} ({dst.stat().st_size / (1024**2):.2f} MB)")
    return dst


def download_all_models(repo_release_url: str = "") -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("VF-50 PRE-LABEL TOOL - MODEL WEIGHTS DOWNLOADER")
    print(f"Target Directory: {MODELS_DIR}")
    print("=" * 70)

    # 1. YuNet (Fast face detector, 232 KB)
    print("\n[1/6] Checking YuNet Face Detector (Level 1 & fallback)...")
    download_with_mirrors(
        [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        MODELS_DIR / "face_detection_yunet_2023mar.onnx",
        min_bytes=200_000,
    )

    # 2. SCRFD-10G (SOTA IR face detector, 16.9 MB)
    print("\n[2/6] Checking SCRFD-10G Detector (Level 2 & Level 3)...")
    download_with_mirrors(
        [
            "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx",
            "https://huggingface.co/Cantina/antelopev2/resolve/main/scrfd_10g_bnkps.onnx",
        ],
        MODELS_DIR / "scrfd_10g_bnkps.onnx",
        min_bytes=15_000_000,
    )

    # 3. SPIGA Base WFLW (254.8 MB)
    print("\n[3/6] Checking SPIGA Base Model (Level 1 & Level 2)...")
    download_with_mirrors(
        [
            "https://huggingface.co/aprados/spiga/resolve/main/spiga_wflw.pt?download=true",
            "https://huggingface.co/aprados/spiga/raw/main/spiga_wflw.pt",
        ],
        MODELS_DIR / "spiga_wflw.pt",
        min_bytes=200_000_000,
    )

    # 4. 6DRepNet Head Pose (157.3 MB)
    print("\n[4/6] Checking 6DRepNet Head Pose Estimator (Level 2 & Level 3)...")
    download_with_mirrors(
        [
            "https://huggingface.co/Cantina/head-pose-estimation/resolve/main/6DRepNet_300W_LP_AFLW2000.pth",
            "https://huggingface.co/BooBooWu/Vec2Face/resolve/main/weights/6DRepNet_300W_LP_AFLW2000.pth",
        ],
        MODELS_DIR / "6DRepNet_300W_LP_AFLW2000.pth",
        min_bytes=100_000_000,
    )

    # 5. TUFA Transformer Base & Shape (Level 2)
    print("\n[5/6] Checking TUFA Landmark Refiner (Level 2 Ensemble)...")
    download_with_mirrors(
        ["https://raw.githubusercontent.com/Jiahao-UTS/TUFA/main/Prompt/shape_98.npz"],
        MODELS_DIR / "tufa_shape_98.npz",
        min_bytes=2_000,
    )
    download_gdrive("1rRAB3W74QK-J6W_EnjC7HhS-zjT9BuED", MODELS_DIR / "tufa_face_prompt_base.pth", min_bytes=100_000_000)

    # 6. SPIGA-IR Fine-tuned Weights (Level 3 - VF-50 IR SOTA)
    print("\n[6/6] Checking Fine-tuned SPIGA-IR Weights (Level 3)...")
    ir_weights = MODELS_DIR / "spiga_wflw_ir_vf50.pt"
    if ir_weights.exists() and ir_weights.stat().st_size >= 200_000_000:
        print(f"  [OK] {ir_weights.name} already present ({ir_weights.stat().st_size / (1024**2):.2f} MB)")
    else:
        release_urls = []
        if repo_release_url:
            release_urls.append(repo_release_url)
        release_urls.extend([
            "https://github.com/Zeus-AIE/vf50-prelabel-tool/releases/download/v1.0.0/spiga_wflw_ir_vf50.pt",
        ])
        download_with_mirrors(release_urls, ir_weights, min_bytes=200_000_000, optional=True)

    print("\n" + "=" * 70)
    print("VERIFYING MODEL INTEGRITY...")
    verify_models()
    print("=" * 70)
    print("[ALL DONE] All models ready for inference!")


def verify_models() -> None:
    # Verify ONNX models
    yunet = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    if yunet.exists():
        sess_yu = ort.InferenceSession(str(yunet), providers=["CPUExecutionProvider"])
        print(f"  [VERIFIED] YuNet detector: OK ({len(sess_yu.get_outputs())} outputs)")

    scrfd = MODELS_DIR / "scrfd_10g_bnkps.onnx"
    if scrfd.exists():
        sess_sc = ort.InferenceSession(str(scrfd), providers=["CPUExecutionProvider"])
        assert len(sess_sc.get_outputs()) == 9
        print(f"  [VERIFIED] SCRFD-10G detector: OK (9 multi-scale outputs)")

    # Verify PyTorch checkpoints
    spiga_base = MODELS_DIR / "spiga_wflw.pt"
    if spiga_base.exists():
        sd = torch.load(str(spiga_base), map_location="cpu", weights_only=False)
        print(f"  [VERIFIED] SPIGA Base weights: OK")

    rep = MODELS_DIR / "6DRepNet_300W_LP_AFLW2000.pth"
    if rep.exists():
        sd_rep = torch.load(str(rep), map_location="cpu", weights_only=False)
        print(f"  [VERIFIED] 6DRepNet weights: OK ({len(sd_rep)} keys)")

    tufa = MODELS_DIR / "tufa_face_prompt_base.pth"
    if tufa.exists():
        sd_tufa = torch.load(str(tufa), map_location="cpu", weights_only=False)
        print(f"  [VERIFIED] TUFA Base weights: OK")

    shape = MODELS_DIR / "tufa_shape_98.npz"
    if shape.exists():
        npz = np.load(str(shape))
        assert npz["offset"].shape == (98, 2)
        print(f"  [VERIFIED] TUFA shape_98: OK (offset shape: {npz['offset'].shape})")

    ir_weights = MODELS_DIR / "spiga_wflw_ir_vf50.pt"
    if ir_weights.exists():
        sd_ir = torch.load(str(ir_weights), map_location="cpu", weights_only=False)
        print(f"  [VERIFIED] SPIGA-IR Fine-tuned weights: OK (Level 3 ready)")
    else:
        print(f"  [NOTE] spiga_wflw_ir_vf50.pt not found. Level 1 & 2 will operate with SPIGA Base.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all pretrained models for VF-50 tool.")
    parser.add_argument("--release-url", default="", help="Custom GitHub release URL for spiga_wflw_ir_vf50.pt")
    args = parser.parse_args()
    download_all_models(repo_release_url=args.release_url)
