"""Download WFLW original uncropped images via HuggingFace CUHK-CSE/wider_face archives."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Set, Tuple

from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"
ARCHIVES = (
    ("WIDER_val.zip", "WIDER_val/images/"),
    ("WIDER_train.zip", "WIDER_train/images/"),
)

ROOT_DIR = Path(__file__).resolve().parent.parent
WFLW_DIR = ROOT_DIR / "datasets" / "wflw"
ANN_DIR = ROOT_DIR / "vendor" / "SPIGA" / "spiga" / "data" / "annotations" / "wflw"


def collect_needed_relpaths() -> Set[str]:
    """Collect relative paths (e.g., '51--Dresses/51_Dresses_...jpg') needed for WFLW train+test."""
    needed: Set[str] = set()
    for split in ("train.json", "test.json"):
        with open(ANN_DIR / split, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            rel = rec["imgpath"]
            if rel.startswith("images/"):
                rel = rel[len("images/") :]
            needed.add(rel)
    return needed


def verify_images(base_dir: Path) -> Tuple[int, int]:
    """Verify all train.json (7500) and test.json (2500) images exist on disk."""
    missing_counts = []
    for split in ("train.json", "test.json"):
        with open(ANN_DIR / split, "r", encoding="utf-8") as f:
            records = json.load(f)
        missing = sum(1 for rec in records if not (base_dir / rec["imgpath"]).exists())
        print(f"[{split}] total={len(records)} missing={missing}")
        missing_counts.append(missing)
    return missing_counts[0], missing_counts[1]


def download_with_resume(url: str, dst: Path) -> None:
    """Download file with HTTP Range resume support and tqdm progress bar."""
    head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(head_req, timeout=30) as hr:
        total_size = int(hr.headers.get("Content-Length", 0))

    existing = dst.stat().st_size if dst.exists() else 0
    if total_size > 0 and existing == total_size:
        print(f"Already downloaded: {dst.name} ({total_size / 1e6:.1f} MB)")
        return
    if existing > total_size > 0:
        dst.unlink()
        existing = 0

    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"Resuming {dst.name} from {existing / 1e6:.1f} / {total_size / 1e6:.1f} MB")
    else:
        print(f"Downloading {dst.name} ({total_size / 1e6:.1f} MB)...")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst, mode) as out_f:
        if existing > 0 and resp.status != 206:
            # Server ignored Range; restart cleanly
            out_f.close()
            dst.unlink()
            return download_with_resume(url, dst)

        with tqdm(
            total=total_size,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=dst.name,
            mininterval=2.0,
        ) as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
                pbar.update(len(chunk))


def extract_needed_from_zip(zip_path: Path, prefix: str, needed: Set[str], out_images_dir: Path) -> int:
    """Extract only the WFLW-referenced images from a WIDER zip archive."""
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix) :]
            if rel not in needed:
                continue
            target = out_images_dir / rel
            if target.exists() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src_f, open(target, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
            extracted += 1
    return extracted


def main() -> None:
    WFLW_DIR.mkdir(parents=True, exist_ok=True)
    out_images = WFLW_DIR / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    miss_tr, miss_te = verify_images(WFLW_DIR)
    if miss_tr == 0 and miss_te == 0:
        print("[PASS] All 7,500 train and 2,500 test WFLW images already present!")
        return

    needed = collect_needed_relpaths()
    print(f"Unique WFLW images needed across train+test: {len(needed)}")

    downloaded_zips = []
    for zip_name, prefix in ARCHIVES:
        zip_path = WFLW_DIR / zip_name
        url = f"{BASE_URL}/{zip_name}"
        download_with_resume(url, zip_path)
        downloaded_zips.append(zip_path)
        cnt = extract_needed_from_zip(zip_path, prefix, needed, out_images)
        print(f"Extracted {cnt} WFLW images from {zip_name}")

    miss_tr, miss_te = verify_images(WFLW_DIR)
    if miss_tr > 0 or miss_te > 0:
        raise RuntimeError(f"Missing WFLW images after extraction: train={miss_tr}, test={miss_te}")

    # Remove large zip archives to save ~1.8 GB disk space after full verification
    for zp in downloaded_zips:
        if zp.exists():
            zp.unlink()
            print(f"Removed temporary archive {zp.name}")

    print("[PASS] All 7,500 train and 2,500 test WFLW images verified!")


if __name__ == "__main__":
    main()
