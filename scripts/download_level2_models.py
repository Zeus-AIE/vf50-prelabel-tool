"""Download and verify Level 2 pretrained weights and clean inference architectures."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

import gdown
import numpy as np
import onnxruntime as ort
import requests
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
VENDOR_TUFA = ROOT_DIR / "vendor" / "TUFA"
VENDOR_6D = ROOT_DIR / "vendor" / "SixDRepNet"


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


def download_with_mirrors(urls: Iterable[str], dst: Path, min_bytes: int) -> Path:
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"[SKIP] {dst.name} already present ({dst.stat().st_size / (1024**2):.2f} MB)")
        return dst

    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"[GET] Downloading {dst.name} from {url} ...")
            _download_http(url, dst)
            size = dst.stat().st_size
            if size < min_bytes:
                raise RuntimeError(f"{dst.name} size too small: {size} < {min_bytes}")
            print(f"[OK] {dst.name} ({size / (1024**2):.2f} MB)")
            return dst
        except Exception as e:
            last_err = e
            print(f"[WARN] Failed {url}: {e}")
    raise RuntimeError(f"Failed downloading {dst.name}: {last_err}")


def download_tufa_weights(dst: Path, min_bytes: int = 100_000_000) -> Path:
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"[SKIP] {dst.name} already present ({dst.stat().st_size / (1024**2):.2f} MB)")
        return dst

    file_id = "1rRAB3W74QK-J6W_EnjC7HhS-zjT9BuED"
    direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        return download_with_mirrors([direct_url], dst, min_bytes=min_bytes)
    except Exception as e:
        print(f"[WARN] Direct Google Drive fetch failed ({e}), trying gdown...")
        dst.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=file_id, output=str(dst), quiet=False)
        if not dst.exists() or dst.stat().st_size < min_bytes:
            raise RuntimeError(f"gdown failed for {dst.name}")
        return dst


def _patch_compat(text: str) -> str:
    trunc_patch = (
        "try:\n"
        "    from timm.layers import trunc_normal_\n"
        "except ImportError:\n"
        "    from timm.models.layers import trunc_normal_"
    )
    text = text.replace("from timm.models.layers import trunc_normal_", trunc_patch)
    text = re.sub(r"\bnp\.float\b", "np.float64", text)
    text = re.sub(r"\bnp\.int\b", "np.int64", text)
    return text


def fetch_vendor_sources() -> None:
    # 1. TUFA inference files (clean subset without training Loss deps)
    tufa_files: Tuple[Tuple[str, Path], ...] = (
        ("https://raw.githubusercontent.com/Jiahao-UTS/TUFA/main/Prompt.py", VENDOR_TUFA / "Prompt.py"),
        ("https://raw.githubusercontent.com/Jiahao-UTS/TUFA/main/Backbone/DINO.py", VENDOR_TUFA / "Backbone" / "DINO.py"),
        ("https://raw.githubusercontent.com/Jiahao-UTS/TUFA/main/Backbone/Decoder.py", VENDOR_TUFA / "Backbone" / "Decoder.py"),
        ("https://raw.githubusercontent.com/Jiahao-UTS/TUFA/main/Backbone/pos_embedding.py", VENDOR_TUFA / "Backbone" / "pos_embedding.py"),
    )
    (VENDOR_TUFA / "Backbone").mkdir(parents=True, exist_ok=True)
    for url, dst in tufa_files:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        txt = _patch_compat(r.text)
        if dst.name == "Prompt.py":
            txt = txt.replace(
                "from Backbone import",
                "from .Backbone import",
            )
        dst.write_text(txt, encoding="utf-8")
        print(f"[VENDOR] Saved {dst}")

    (VENDOR_TUFA / "__init__.py").write_text(
        '"""Clean TUFA inference package."""\n', encoding="utf-8"
    )
    (VENDOR_TUFA / "Backbone" / "__init__.py").write_text(
        "from .Decoder import Decoder\n"
        "from .pos_embedding import Gen_2D_embedding_from_vector\n"
        "from .DINO import load_DINO\n",
        encoding="utf-8",
    )

    # 2. SixDRepNet backbone files
    VENDOR_6D.mkdir(parents=True, exist_ok=True)
    for fname in ("repvgg.py", "se_block.py"):
        url = f"https://raw.githubusercontent.com/thohemp/6DRepNet/master/sixdrepnet/backbone/{fname}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        txt = _patch_compat(r.text)
        if fname == "repvgg.py":
            txt = txt.replace("from backbone.se_block import SEBlock", "from .se_block import SEBlock")
        (VENDOR_6D / fname).write_text(txt, encoding="utf-8")
        print(f"[VENDOR] Saved {VENDOR_6D / fname}")

    (VENDOR_6D / "__init__.py").write_text(
        "from .repvgg import get_RepVGG_func_by_name\n", encoding="utf-8"
    )


def verify_all() -> None:
    scrfd_path = MODELS_DIR / "scrfd_10g_bnkps.onnx"
    rep_path = MODELS_DIR / "6DRepNet_300W_LP_AFLW2000.pth"
    tufa_path = MODELS_DIR / "tufa_face_prompt_base.pth"
    shape_path = MODELS_DIR / "tufa_shape_98.npz"

    sess = ort.InferenceSession(str(scrfd_path), providers=["CPUExecutionProvider"])
    outs = sess.get_outputs()
    print(f"[VERIFY] SCRFD-10G ONNX outputs={len(outs)}, inputs={[(i.name, i.shape) for i in sess.get_inputs()]}")
    assert len(outs) == 9, f"Unexpected SCRFD output count: {len(outs)}"

    sd_rep = torch.load(str(rep_path), map_location="cpu", weights_only=False)
    print(f"[VERIFY] 6DRepNet keys={len(sd_rep)}")

    sd_tufa = torch.load(str(tufa_path), map_location="cpu", weights_only=False)
    print(f"[VERIFY] TUFA weights type={type(sd_tufa).__name__}, keys={len(sd_tufa)}")

    npz = np.load(str(shape_path))
    offset = npz["offset"]
    print(f"[VERIFY] TUFA shape_98 offset shape={offset.shape}")
    assert offset.shape == (98, 2)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_with_mirrors(
        [
            "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx",
            "https://huggingface.co/Cantina/antelopev2/resolve/main/scrfd_10g_bnkps.onnx",
        ],
        MODELS_DIR / "scrfd_10g_bnkps.onnx",
        min_bytes=15_000_000,
    )
    download_with_mirrors(
        [
            "https://huggingface.co/Cantina/head-pose-estimation/resolve/main/6DRepNet_300W_LP_AFLW2000.pth",
            "https://huggingface.co/BooBooWu/Vec2Face/resolve/main/weights/6DRepNet_300W_LP_AFLW2000.pth",
        ],
        MODELS_DIR / "6DRepNet_300W_LP_AFLW2000.pth",
        min_bytes=100_000_000,
    )
    download_tufa_weights(MODELS_DIR / "tufa_face_prompt_base.pth", min_bytes=100_000_000)
    download_with_mirrors(
        ["https://raw.githubusercontent.com/Jiahao-UTS/TUFA/main/Prompt/shape_98.npz"],
        MODELS_DIR / "tufa_shape_98.npz",
        min_bytes=2_000,
    )
    fetch_vendor_sources()
    verify_all()
    print("[DONE] All Level 2 weights and vendor architectures verified!")


if __name__ == "__main__":
    main()
