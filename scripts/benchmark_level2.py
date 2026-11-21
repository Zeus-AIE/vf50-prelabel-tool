"""Benchmark Level 2 (SPIGA-IR, Flip TTA, TUFA, Ensemble weights, 6DRepNet, 2-Pass) on 500 WFLW test faces."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.headpose_6drepnet import SixDRepNetEngine
from src.ir_augment import SyntheticIRAugmentor
from src.landmark_engine import EnsembleVF50Engine, VF50Engine
from src.tufa_engine import TUFAEngine
from src.vf50_schema import VF50_TO_WFLW98

ANN_DIR = ROOT_DIR / "vendor" / "SPIGA" / "spiga" / "data" / "annotations" / "wflw"
WFLW_DIR = ROOT_DIR / "datasets" / "wflw"
OUT_JSON = ROOT_DIR / "models" / "level2_benchmark.json"
VF50_IDX = np.array(VF50_TO_WFLW98, dtype=np.int64)


def select_representative_500(test_recs: List[dict], db_info: dict) -> Tuple[List[int], Dict[str, set]]:
    """Select 500 indices stratified across pose, blur, occlusion, illumination, and general."""
    subsets = db_info.get("test_subsets", {})
    picked: List[int] = []
    seen = set()

    # Take up to 90 from each hard subset cleanly interleaved
    for key in ("pose", "blur", "occlusion", "illumination"):
        sub = subsets.get(key, [])
        step = max(1, len(sub) // 90)
        for idx in sub[::step]:
            if idx not in seen and len(picked) < 360:
                picked.append(idx)
                seen.add(idx)

    # Fill remaining up to 500 deterministically across all 2500
    for idx in range(0, len(test_recs), 5):
        if idx not in seen and len(picked) < 500:
            picked.append(idx)
            seen.add(idx)

    picked.sort()
    sub_sets = {k: set(subsets.get(k, [])) for k in ("pose", "blur", "occlusion", "illumination")}
    return picked, sub_sets


def make_roi_and_ir(
    img_bgr: np.ndarray, bbox: List[float], gt_98: np.ndarray, rng: np.random.RandomState, roi_sz: int = 320
) -> Tuple[np.ndarray, np.ndarray, List[float], np.ndarray]:
    """Create a realistic padded cabin-resolution (320x320) Clean + Synthetic IR pair with exact transformed GT."""
    bx, by, bw, bh = [float(v) for v in bbox]
    cx, cy = bx + 0.5 * bw, by + 0.5 * bh
    side = max(bw, bh, 32.0) * 2.4
    scale = float(roi_sz) / side
    mat = np.array(
        [
            [scale, 0.0, 0.5 * roi_sz - scale * cx],
            [0.0, scale, 0.5 * roi_sz - scale * cy],
        ],
        dtype=np.float64,
    )
    clean_roi = cv2.warpAffine(
        img_bgr, mat.astype(np.float32), (roi_sz, roi_sz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    gt_h = np.concatenate([gt_98, np.ones((len(gt_98), 1), dtype=np.float64)], axis=1)
    gt_roi = (mat @ gt_h.T).T

    # Slightly jitter bbox like detector output doesn't cheat, or use transformed WFLW bbox
    x1, y1 = (mat @ np.array([bx, by, 1.0])).tolist()
    x2, y2 = (mat @ np.array([bx + bw, by + bh, 1.0])).tolist()
    bbox_roi = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    ir_aug = SyntheticIRAugmentor(prob=1.0, glasses_prob=0.35, occlusion_prob=0.25)
    ir_roi = ir_aug(clean_roi, landmarks_256=gt_roi, rng=rng)
    return clean_roi, ir_roi, bbox_roi, gt_roi


def calc_nme(pred_98: np.ndarray, gt_98: np.ndarray) -> Tuple[float, float]:
    iod = float(np.linalg.norm(gt_98[60] - gt_98[72]))
    if iod < 1e-4:
        return 0.0, 0.0
    d = np.linalg.norm(pred_98 - gt_98, axis=1)
    return float(np.mean(d[VF50_IDX]) / iod * 100.0), float(np.mean(d) / iod * 100.0)


def main() -> None:
    with open(ANN_DIR / "test.json", "r", encoding="utf-8") as f:
        test_recs = json.load(f)
    with open(ANN_DIR / "db_info.json", "r", encoding="utf-8") as f:
        db_info = json.load(f)

    picked, sub_sets = select_representative_500(test_recs, db_info)
    print(f"[DATA] Benchmarking on {len(picked)} stratified WFLW test images...", flush=True)

    spiga = VF50Engine(str(ROOT_DIR / "models"), device="cuda", weights_name="spiga_wflw_ir_vf50.pt")
    tufa = TUFAEngine(
        str(ROOT_DIR / "models" / "tufa_face_prompt_base.pth"),
        str(ROOT_DIR / "models" / "tufa_shape_98.npz"),
        device="cuda",
    )
    rep = SixDRepNetEngine(str(ROOT_DIR / "models" / "6DRepNet_300W_LP_AFLW2000.pth"), device="cuda")
    torch.cuda.reset_peak_memory_stats(0)

    methods = (
        "spiga_ir_1pass",
        "spiga_ir_tta",
        "tufa_tta",
        "ens_w50_50",
        "ens_w60_40",
        "ens_w65_35",
    )
    results: Dict[str, Dict[str, List[float]]] = {
        domain: {m: [] for m in methods} for domain in ("clean", "ir")
    }
    subset_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        domain: {sub: {m: [] for m in methods} for sub in sub_sets} for domain in ("clean", "ir")
    }

    rng = np.random.RandomState(2026)
    for k, rec_idx in enumerate(picked):
        rec = test_recs[rec_idx]
        im = cv2.imread(str(WFLW_DIR / rec["imgpath"]))
        gt_orig = np.array(rec["landmarks"], dtype=np.float64)
        clean_roi, ir_roi, bbox_roi, gt_roi = make_roi_and_ir(im, rec["bbox"], gt_orig, rng)

        for domain, img_in in (("clean", clean_roi), ("ir", ir_roi)):
            p_sp1, _, _, _, _ = spiga.predict_wflw98_tta(img_in, bbox_roi, use_flip_tta=False)
            p_spt, _, _, _, _ = spiga.predict_wflw98_tta(img_in, bbox_roi, use_flip_tta=True)
            p_tuf, _ = tufa.predict_wflw98(img_in, bbox_roi, use_flip_tta=True)

            iod = max(float(np.linalg.norm(p_spt[60] - p_spt[72])), 16.0)
            outlier = float(np.median(np.linalg.norm(p_spt - p_tuf, axis=1))) > 0.25 * iod

            preds = {
                "spiga_ir_1pass": p_sp1,
                "spiga_ir_tta": p_spt,
                "tufa_tta": p_tuf,
                "ens_w50_50": p_spt if outlier else (0.50 * p_spt + 0.50 * p_tuf),
                "ens_w60_40": p_spt if outlier else (0.60 * p_spt + 0.40 * p_tuf),
                "ens_w65_35": p_spt if outlier else (0.65 * p_spt + 0.35 * p_tuf),
            }
            for m, pred98 in preds.items():
                nme50, _ = calc_nme(pred98, gt_roi)
                results[domain][m].append(nme50)
                for sub_name, id_set in sub_sets.items():
                    if rec_idx in id_set:
                        subset_scores[domain][sub_name][m].append(nme50)

        if (k + 1) % 100 == 0 or (k + 1) == len(picked):
            print(
                f"  [{k+1}/{len(picked)}] IR NME: spiga_1p={np.mean(results['ir']['spiga_ir_1pass']):.3f}% | "
                f"spiga_tta={np.mean(results['ir']['spiga_ir_tta']):.3f}% | "
                f"tufa_tta={np.mean(results['ir']['tufa_tta']):.3f}% | "
                f"ens60={np.mean(results['ir']['ens_w60_40']):.3f}%",
                flush=True,
            )

    # Full Level-2 E2E timing + 2-Pass evaluation on first 100 IR ROIs (including warmup)
    ens_engine = EnsembleVF50Engine(spiga, tufa_engine=tufa, headpose_engine=rep, spiga_weight=0.60)
    rng_t = np.random.RandomState(2026)
    e2e_nme50 = []
    # Warmup 3 images
    for rec_idx in picked[:3]:
        rec = test_recs[rec_idx]
        im = cv2.imread(str(WFLW_DIR / rec["imgpath"]))
        _, ir_roi, bbox_roi, _ = make_roi_and_ir(im, rec["bbox"], np.array(rec["landmarks"], dtype=np.float64), np.random.RandomState(rec_idx))
        ens_engine.predict(ir_roi, bbox_roi, use_tta=True, use_two_pass=True, use_ensemble=True, use_6drepnet=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for rec_idx in picked[:100]:
        rec = test_recs[rec_idx]
        im = cv2.imread(str(WFLW_DIR / rec["imgpath"]))
        _, ir_roi, bbox_roi, gt_roi = make_roi_and_ir(im, rec["bbox"], np.array(rec["landmarks"], dtype=np.float64), rng_t)
        pred = ens_engine.predict(ir_roi, bbox_roi, use_tta=True, use_two_pass=True, use_ensemble=True, use_6drepnet=True)
        nme50, _ = calc_nme(pred.points_wflw98, gt_roi)
        e2e_nme50.append(nme50)
    torch.cuda.synchronize()
    ms_per_img = (time.perf_counter() - t0) * 1000.0 / 100.0
    peak_vram_mib = torch.cuda.max_memory_allocated(0) / (1024**2)

    summary = {
        "num_images": len(picked),
        "clean_vf50_nme": {m: round(float(np.mean(results["clean"][m])), 4) for m in methods},
        "ir_vf50_nme": {m: round(float(np.mean(results["ir"][m])), 4) for m in methods},
        "ir_subsets_vf50_nme": {
            sub: {m: round(float(np.mean(subset_scores["ir"][sub][m])), 4) for m in methods}
            for sub in sub_sets
        },
        "full_level2_2pass_100ir_vf50_nme": round(float(np.mean(e2e_nme50)), 4),
        "full_level2_ms_per_img": round(ms_per_img, 1),
        "peak_vram_mib": round(peak_vram_mib, 1),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== LEVEL 2 BENCHMARK SUMMARY (500 WFLW Stratified Test Images) ===", flush=True)
    for m in methods:
        print(f"  {m:16s}: Clean={summary['clean_vf50_nme'][m]:.3f}% | IR={summary['ir_vf50_nme'][m]:.3f}%", flush=True)
    print(f"[FULL LEVEL 2 2-Pass+TTA+Ensemble+6DRepNet] 100-IR VF50={summary['full_level2_2pass_100ir_vf50_nme']:.3f}% | Speed={ms_per_img:.1f} ms/img | PeakVRAM={peak_vram_mib:.1f} MiB", flush=True)
    print(f"[SAVED] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
