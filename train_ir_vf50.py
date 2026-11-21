"""Fine-tune SPIGA WFLW-98 for VF-50 facial landmarks on Synthetic Cabin IR + WFLW."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "vendor" / "SPIGA"))

from spiga.data.loaders.augmentors.landmarks import HorizontalFlipAug, RSTAug, TargetCropAug
from spiga.data.loaders.augmentors.modern_posit import PositPose
from spiga.data.loaders.dl_config import AlignConfig, DatabaseStruct
from spiga.data.loaders.transforms import ToOpencv
from spiga.inference.config import ModelConfig
from spiga.models.spiga import SPIGA

from src.ir_augment import SyntheticIRAugmentor
from src.vf50_schema import VF50_TO_WFLW98

ANN_DIR = ROOT_DIR / "vendor" / "SPIGA" / "spiga" / "data" / "annotations" / "wflw"
DEFAULT_WFLW_DIR = ROOT_DIR / "datasets" / "wflw"
DEFAULT_BASE_WEIGHTS = ROOT_DIR / "models" / "spiga_wflw.pt"
DEFAULT_OUT_WEIGHTS = ROOT_DIR / "models" / "spiga_wflw_ir_vf50.pt"
DEFAULT_REPORT_PATH = ROOT_DIR / "models" / "ir_finetune_report.json"
DEFAULT_PROGRESS_PATH = ROOT_DIR / "models" / "train_progress.json"

# 98-dim weight vector: 2.0x for VF-50 points, 1.0x for remaining jawline/contour points
VF50_SET = set(VF50_TO_WFLW98)
VF50_INDEX_ARRAY = np.array(VF50_TO_WFLW98, dtype=np.int64)


class WFLWIRDataset(Dataset):
    """WFLW-98 dataset with SPIGA geometric pipeline + SyntheticIRAugmentor."""

    def __init__(
        self,
        records: List[dict],
        images_root: Path,
        database: DatabaseStruct,
        posit_ids: List[int],
        is_train: bool = True,
        ir_prob: float = 0.85,
        deterministic_ir: bool = False,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.images_root = images_root
        self.database = database
        self.is_train = is_train
        self.deterministic_ir = deterministic_ir
        self.seed = seed

        self.hflip = HorizontalFlipAug(database.ldm_flip_order, prob=0.5)
        self.rst = RSTAug(angle_range=35.0, scale_min=-0.15, scale_max=0.15, trl_ratio=0.05)
        self.crop = TargetCropAug((256, 256), (64, 64), target_dist=1.6)
        self.to_cv = ToOpencv()
        self.posit = PositPose(
            database.ldm_ids,
            focal_ratio=1.5,
            selected_ids=posit_ids,
            max_iter=30,
        )
        self.ir_aug = SyntheticIRAugmentor(prob=ir_prob, glasses_prob=0.30, occlusion_prob=0.25)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        img_path = self.images_root / rec["imgpath"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if bgr.ndim == 2:
            bgr = np.repeat(bgr[:, :, None], 3, axis=2)
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        landmarks = np.array(rec["landmarks"], dtype=np.float64)
        bbox = np.array(rec["bbox"], dtype=np.float64)
        vis = np.array(rec["visible"], dtype=np.float64)
        mask_ldm = np.ones(self.database.num_landmarks, dtype=np.float64)

        sample = {
            "image": pil_img,
            "sample_idx": idx,
            "ids_ldm": np.array(self.database.ldm_ids),
            "bbox": bbox.copy(),
            "bbox_raw": bbox.copy(),
            "landmarks": landmarks.copy(),
            "visible": vis.copy(),
            "mask_ldm": mask_ldm,
        }

        if self.is_train:
            sample = self.hflip(sample)
            sample = self.rst(sample)

        sample = self.crop(sample)
        sample = self.to_cv(sample)
        if self.is_train:
            sample = self.posit(sample)
        else:
            img_shape = np.array(sample["image"].shape)[0:2]
            if "img2map_scale" in sample:
                img_shape = img_shape * sample["img2map_scale"]
            img_bbox = [0, 0, img_shape[1], img_shape[0]]
            sample["cam_matrix"] = self.posit._camera_matrix(img_bbox)
            sample["model3d"] = self.posit.model3d_world
            sample["pose"] = np.zeros(6, dtype=np.float32)

        img_bgr = sample["image"]  # 256x256x3 uint8 BGR
        lms_64 = sample["landmarks_float"].astype(np.float32)  # (98, 2) in [0, 64]
        lms_256 = lms_64 * 4.0

        if self.ir_aug.prob > 0.0:
            if self.deterministic_ir:
                rng = np.random.RandomState(self.seed + idx * 9973)
            else:
                rng = np.random.RandomState(random.randint(0, 2**31 - 1))
            img_bgr = self.ir_aug(img_bgr, landmarks_256=lms_256, rng=rng)

        img_chw = np.transpose(img_bgr.astype(np.float32), (2, 0, 1)) / 255.0
        lms_norm = np.clip(lms_64 / 64.0, 0.0, 1.0)
        mask = sample["mask_ldm"].astype(np.float32)

        return {
            "image": torch.from_numpy(img_chw),
            "landmarks": torch.from_numpy(lms_norm),
            "mask": torch.from_numpy(mask),
            "pose": torch.from_numpy(sample["pose"].astype(np.float32)),
            "model3d": torch.from_numpy(sample["model3d"].astype(np.float32)),
            "cam_matrix": torch.from_numpy(sample["cam_matrix"].astype(np.float32)),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def split_train_val_by_image(records: List[dict], val_ratio: float = 0.10, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Group-split by imgpath so faces from the same photo never cross train/val."""
    unique_imgs = sorted({r["imgpath"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(unique_imgs)
    n_val_imgs = max(1, int(round(len(unique_imgs) * val_ratio)))
    val_imgs = set(unique_imgs[:n_val_imgs])
    train_recs = [r for r in records if r["imgpath"] not in val_imgs]
    val_recs = [r for r in records if r["imgpath"] in val_imgs]
    return train_recs, val_recs


def compute_sample_weights(records: List[dict]) -> List[float]:
    """Upweight strong profile faces (eye width asymmetry > 1.4x) by 2.5x."""
    weights: List[float] = []
    for rec in records:
        lms = np.array(rec["landmarks"], dtype=np.float64)
        eye_l_w = abs(lms[64, 0] - lms[60, 0])
        eye_r_w = abs(lms[72, 0] - lms[68, 0])
        min_w = max(min(eye_l_w, eye_r_w), 0.5)
        max_w = max(eye_l_w, eye_r_w)
        ratio = max_w / min_w
        weights.append(2.5 if ratio > 1.4 else 1.0)
    return weights


def build_spiga_model(weights_path: Path, device: torch.device) -> SPIGA:
    cfg = ModelConfig("wflw", load_model_url=False)
    model = SPIGA(
        num_landmarks=cfg.dataset.num_landmarks,
        num_edges=cfg.dataset.num_edges,
    )
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    gt_lms: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_pose: torch.Tensor,
    pt_weights: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Multi-step SmoothL1 weighted 2.0x on VF-50 landmarks + auxiliary 6DoF pose loss."""
    step_weights = (0.25, 0.50, 1.00)
    smooth_l1 = nn.SmoothL1Loss(beta=0.015, reduction="none")

    total_ldm_loss = torch.tensor(0.0, device=gt_lms.device)
    # mask shape: (B, 98, 1), pt_weights: (1, 98, 1)
    eff_mask = gt_mask.unsqueeze(-1) * pt_weights
    denom = eff_mask.sum().clamp_min(1.0) * 2.0

    for sw, pred_step in zip(step_weights, outputs["Landmarks"]):
        elem_loss = smooth_l1(pred_step.float(), gt_lms) * eff_mask
        total_ldm_loss = total_ldm_loss + sw * (elem_loss.sum() / denom)

    pose_loss = nn.SmoothL1Loss(beta=1.0)(outputs["Pose"].float(), gt_pose)
    total = total_ldm_loss + 0.05 * pose_loss
    return total, {"ldm": float(total_ldm_loss.item()), "pose": float(pose_loss.item())}


@torch.no_grad()
def evaluate_dataset(
    model: SPIGA,
    loader: DataLoader,
    device: torch.device,
    records: Optional[List[dict]] = None,
    subset_indices: Optional[Dict[str, Sequence[int]]] = None,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """Compute inter-ocular NME% for WFLW-98 and VF-50 (overall + optional test subsets)."""
    model.eval()
    nme_98_list: List[float] = []
    nme_50_list: List[float] = []
    sample_indices: List[int] = []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        model3d = batch["model3d"][0].to(device, non_blocking=True)
        cam_matrix = batch["cam_matrix"][0].to(device, non_blocking=True)
        b_size = imgs.shape[0]
        batch_m3d = model3d.unsqueeze(0).repeat(b_size, 1, 1)
        batch_cam = cam_matrix.unsqueeze(0).repeat(b_size, 1, 1)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            out = model([imgs, batch_m3d, batch_cam])

        preds = out["Landmarks"][-1].float().cpu().numpy()  # (B, 98, 2)
        gts = batch["landmarks"].numpy()
        masks = batch["mask"].numpy()
        idxs = batch["idx"].numpy()

        for b in range(b_size):
            pred = preds[b]
            gt = gts[b]
            m = masks[b] > 0.5
            # WFLW standard inter-ocular distance: outer eye corners 60 & 72
            iod = float(np.linalg.norm(gt[60] - gt[72]))
            if iod < 1e-4:
                continue
            dists = np.linalg.norm(pred - gt, axis=1)
            nme_98 = float(np.mean(dists[m]) / iod * 100.0) if m.any() else 0.0

            m50 = m[VF50_INDEX_ARRAY]
            d50 = dists[VF50_INDEX_ARRAY]
            nme_50 = float(np.mean(d50[m50]) / iod * 100.0) if m50.any() else 0.0

            nme_98_list.append(nme_98)
            nme_50_list.append(nme_50)
            sample_indices.append(int(idxs[b]))

    metrics: Dict[str, float] = {
        "wflw98_nme": round(float(np.mean(nme_98_list)), 4),
        "vf50_nme": round(float(np.mean(nme_50_list)), 4),
        "count": float(len(nme_50_list)),
    }

    if records is not None and subset_indices:
        # Map dataset idx -> computed NME
        nme50_by_idx = {si: n50 for si, n50 in zip(sample_indices, nme_50_list)}
        nme98_by_idx = {si: n98 for si, n98 in zip(sample_indices, nme_98_list)}
        for sub_name, sub_ids in subset_indices.items():
            sub_id_set = set(sub_ids)
            s50 = [nme50_by_idx[i] for i in sample_indices if i in sub_id_set]
            s98 = [nme98_by_idx[i] for i in sample_indices if i in sub_id_set]
            if s50:
                metrics[f"{sub_name}_vf50_nme"] = round(float(np.mean(s50)), 4)
                metrics[f"{sub_name}_wflw98_nme"] = round(float(np.mean(s98)), 4)
    return metrics


def generate_ir_preview(images_root: Path, records: List[dict], out_path: Path) -> None:
    """Create 8-sample side-by-side RGB vs Synthetic IR preview with VF-50 GT landmarks."""
    db = DatabaseStruct("wflw")
    align_cfg = AlignConfig("wflw", mode="test")
    crop_tf = TargetCropAug((256, 256), (64, 64), target_dist=1.6)
    to_cv = ToOpencv()
    ir_aug = SyntheticIRAugmentor(prob=1.0, glasses_prob=0.45, occlusion_prob=0.35)

    # Pick diverse samples (including asymmetric profile faces)
    scored = []
    for idx, r in enumerate(records[:1200]):
        lms = np.array(r["landmarks"], dtype=np.float64)
        el = max(abs(lms[64, 0] - lms[60, 0]), 1.0)
        er = max(abs(lms[72, 0] - lms[68, 0]), 1.0)
        asym = max(el, er) / min(el, er)
        scored.append((asym, idx))
    scored.sort(reverse=True)
    chosen_idxs = [x[1] for x in scored[:4]] + [10, 55, 120, 210]

    tiles = []
    rng = np.random.RandomState(2026)
    for idx in chosen_idxs:
        rec = records[idx]
        bgr = cv2.imread(str(images_root / rec["imgpath"]))
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        sample = {
            "image": Image.fromarray(rgb),
            "bbox": np.array(rec["bbox"], dtype=np.float64),
            "landmarks": np.array(rec["landmarks"], dtype=np.float64),
            "mask_ldm": np.ones(db.num_landmarks, dtype=np.float64),
        }
        sample = crop_tf(sample)
        sample = to_cv(sample)
        rgb_crop = sample["image"].copy()
        lms_256 = sample["landmarks_float"] * 4.0
        ir_crop = ir_aug(rgb_crop, landmarks_256=lms_256, rng=rng)

        vis_rgb = rgb_crop.copy()
        vis_ir = ir_crop.copy()
        for pid, w_idx in enumerate(VF50_TO_WFLW98):
            x, y = int(round(lms_256[w_idx, 0])), int(round(lms_256[w_idx, 1]))
            cv2.circle(vis_rgb, (x, y), 2, (0, 255, 0), -1)
            cv2.circle(vis_ir, (x, y), 2, (0, 220, 255), -1)
        pair = np.hstack([vis_rgb, vis_ir])
        tiles.append(pair)

    grid = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6]), np.hstack(tiles[6:8])])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[INFO] Saved IR augmentation preview to {out_path}")


def get_amp_dtype(device: torch.device) -> torch.dtype:
    """Use bfloat16 on Ampere+ GPUs (RTX 3060) to avoid GradScaler overhead and improve throughput."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_stage1_freeze(model: SPIGA, orig_trainable_names: Set[str]) -> None:
    """Freeze visual_cnn stem and hourglass stages 0..2; train hgs[3] + pose + GAT heads."""
    for name, p in model.named_parameters():
        if name not in orig_trainable_names:
            p.requires_grad = False
            continue
        if name.startswith("visual_cnn."):
            p.requires_grad = name.startswith(
                ("visual_cnn.hgs.3.", "visual_cnn.hgs_out.3.", "visual_cnn.hgs_core.3.")
            )
        else:
            p.requires_grad = True


def set_stage1_train_mode(model: SPIGA) -> None:
    """Keep frozen visual_cnn submodules in eval() so BatchNorm running stats do not drift."""
    model.train()
    frozen_modules = [
        model.visual_cnn.pre,
        model.visual_cnn.hgs[0],
        model.visual_cnn.hgs[1],
        model.visual_cnn.hgs[2],
        model.visual_cnn.hgs_out[0],
        model.visual_cnn.hgs_out[1],
        model.visual_cnn.hgs_out[2],
        model.visual_cnn.hgs_core[0],
        model.visual_cnn.hgs_core[1],
        model.visual_cnn.hgs_core[2],
        model.visual_cnn.outs_points,
        model.visual_cnn.outs_edges,
        model.visual_cnn.outs_features,
        model.visual_cnn.merge_preds,
        model.visual_cnn.merge_features,
    ]
    for mod in frozen_modules:
        mod.eval()


def set_stage2_unfreeze(model: SPIGA, orig_trainable_names: Set[str]) -> None:
    """Restore original trainability across full SPIGA network without unfreezing non-param buffers."""
    for name, p in model.named_parameters():
        p.requires_grad = name in orig_trainable_names


def run_smoke_test(wflw_dir: Path, base_weights: Path) -> None:
    """Verify coordinate alignment & forward/backward pass on 1 batch before full training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)
    db = DatabaseStruct("wflw")
    align_cfg = AlignConfig("wflw", mode="test")
    with open(ANN_DIR / "test.json", "r", encoding="utf-8") as f:
        test_recs = json.load(f)[:64]

    ds_clean = WFLWIRDataset(test_recs, wflw_dir, db, align_cfg.posit_ids, is_train=False, ir_prob=0.0)
    loader = DataLoader(ds_clean, batch_size=32, shuffle=False, num_workers=0)
    model = build_spiga_model(base_weights, device)
    metrics = evaluate_dataset(model, loader, device, amp_dtype=amp_dtype)
    print(f"[SMOKE TEST] ({amp_dtype}) 64 clean test faces -> WFLW98 NME={metrics['wflw98_nme']:.3f}%, VF50 NME={metrics['vf50_nme']:.3f}%", flush=True)
    if metrics["wflw98_nme"] > 7.5:
        raise RuntimeError(f"Smoke test NME too high ({metrics['wflw98_nme']:.3f}%), coordinate mismatch suspected!")


def clone_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot model weights onto CPU RAM to avoid holding duplicate tensors in VRAM."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def write_progress(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write training progress JSON for external monitoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler: Optional[WeightedRandomSampler] = None,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Construct DataLoader with persistent workers on Windows when num_workers > 0."""
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def run_stage(
    stage_id: int,
    stage_epochs: int,
    stage_bs: int,
    stage_grad_accum: int,
    backbone_lr: float,
    head_lr: float,
    epoch_offset: int,
    total_epochs: int,
    model: SPIGA,
    train_ds: WFLWIRDataset,
    sample_weights: List[float],
    val_clean_loader: DataLoader,
    val_ir_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    pt_weights: torch.Tensor,
    num_workers: int,
    log_every: int,
    out_weights: Path,
    progress_path: Path,
    best_ir_vf50: float,
    best_state: Dict[str, torch.Tensor],
    history: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Run one fine-tuning stage (Stage 1 frozen stem/hg0-2, or Stage 2 full unfreeze)."""
    if stage_epochs <= 0:
        return best_ir_vf50, best_state

    pin_mem = device.type == "cuda"
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = make_dataloader(
        train_ds,
        batch_size=stage_bs,
        num_workers=num_workers,
        pin_memory=pin_mem,
        sampler=sampler,
        drop_last=True,
    )

    backbone_params = [p for p in model.visual_cnn.parameters() if p.requires_grad]
    head_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("visual_cnn.")
    ]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    n_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    print(
        f"\n[STAGE {stage_id}] epochs={stage_epochs} | bs={stage_bs} | grad_accum={stage_grad_accum} | "
        f"backbone_lr={backbone_lr} | head_lr={head_lr} | trainable_params={n_trainable:,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-4,
        fused=(device.type == "cuda"),
    )
    eta_min = 1e-6 if stage_id == 2 else 2e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(stage_epochs, 1), eta_min=eta_min
    )
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    total_steps = len(train_loader)
    for local_ep in range(1, stage_epochs + 1):
        global_ep = epoch_offset + local_ep
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if stage_id == 1:
            set_stage1_train_mode(model)
        else:
            model.train()

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            imgs = batch["image"].to(device, non_blocking=True)
            gt_lms = batch["landmarks"].to(device, non_blocking=True)
            gt_mask = batch["mask"].to(device, non_blocking=True)
            gt_pose = batch["pose"].to(device, non_blocking=True)
            model3d = batch["model3d"][0].to(device, non_blocking=True)
            cam_matrix = batch["cam_matrix"][0].to(device, non_blocking=True)
            b_size = imgs.shape[0]
            batch_m3d = model3d.unsqueeze(0).repeat(b_size, 1, 1)
            batch_cam = cam_matrix.unsqueeze(0).repeat(b_size, 1, 1)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                outputs = model([imgs, batch_m3d, batch_cam])
                loss, _ = compute_loss(outputs, gt_lms, gt_mask, gt_pose, pt_weights)
                loss_scaled = loss / stage_grad_accum

            scaler.scale(loss_scaled).backward()
            running_loss += float(loss.item())
            steps += 1

            if step_idx % stage_grad_accum == 0 or step_idx == total_steps:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], 5.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step_idx == 10 or step_idx % log_every == 0 or step_idx == total_steps:
                elapsed = time.time() - t0
                bps = step_idx / max(elapsed, 1e-6)
                eta = (total_steps - step_idx) / max(bps, 1e-6)
                pct = 100.0 * step_idx / max(total_steps, 1)
                vram_mib = (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0.0
                )
                avg_step_loss = running_loss / max(steps, 1)
                print(
                    f"[STAGE {stage_id} | EPOCH {global_ep}/{total_epochs} | step {step_idx}/{total_steps} ({pct:.1f}%)] "
                    f"loss={avg_step_loss:.5f} | {bps:.2f} batch/s ({bps * stage_bs:.1f} img/s) | "
                    f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s | PeakVRAM={vram_mib:.0f}MiB",
                    flush=True,
                )
                write_progress(
                    progress_path,
                    {
                        "status": f"running_stage{stage_id}",
                        "stage": stage_id,
                        "epoch": global_ep,
                        "total_epochs": total_epochs,
                        "batch_size": stage_bs,
                        "step": step_idx,
                        "total_steps": total_steps,
                        "percent": round(pct, 1),
                        "loss": round(avg_step_loss, 5),
                        "batch_per_sec": round(bps, 3),
                        "elapsed_sec": round(elapsed, 1),
                        "eta_sec": round(eta, 1),
                        "peak_vram_mib": round(vram_mib, 1),
                        "best_ir_vf50_nme": best_ir_vf50,
                        "history": history,
                    },
                )

        scheduler.step()
        val_clean = evaluate_dataset(model, val_clean_loader, device, amp_dtype=amp_dtype)
        val_ir = evaluate_dataset(model, val_ir_loader, device, amp_dtype=amp_dtype)
        dt = time.time() - t0
        avg_loss = running_loss / max(steps, 1)

        improved = ""
        if val_ir["vf50_nme"] < best_ir_vf50:
            best_ir_vf50 = val_ir["vf50_nme"]
            best_state = clone_state_dict_cpu(model)
            out_weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, str(out_weights))
            improved = " *BEST*"

        print(
            f"[STAGE {stage_id} | EPOCH {global_ep}/{total_epochs}] {dt:.1f}s | loss={avg_loss:.5f} | "
            f"Clean Val VF50={val_clean['vf50_nme']:.3f}% | "
            f"IR Val VF50={val_ir['vf50_nme']:.3f}% (WFLW98={val_ir['wflw98_nme']:.3f}%){improved}",
            flush=True,
        )
        history.append(
            {
                "stage": stage_id,
                "epoch": global_ep,
                "epoch_sec": round(dt, 1),
                "loss": round(avg_loss, 5),
                "val_clean": val_clean,
                "val_ir": val_ir,
            }
        )
        write_progress(
            progress_path,
            {
                "status": f"running_stage{stage_id}",
                "stage": stage_id,
                "epoch": global_ep,
                "total_epochs": total_epochs,
                "best_ir_vf50_nme": best_ir_vf50,
                "history": history,
            },
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Release DataLoader persistent workers & optimizer buffers before next stage / test
    del train_loader, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_ir_vf50, best_state


def train_and_evaluate(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = get_amp_dtype(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    wflw_dir = Path(args.wflw_dir)
    base_weights = Path(args.base_weights)
    out_weights = Path(args.out_weights)
    report_path = Path(args.report_path)
    progress_path = Path(args.progress_path)

    with open(ANN_DIR / "train.json", "r", encoding="utf-8") as f:
        all_train_recs = json.load(f)
    with open(ANN_DIR / "test.json", "r", encoding="utf-8") as f:
        test_recs = json.load(f)
    with open(ANN_DIR / "db_info.json", "r", encoding="utf-8") as f:
        db_info = json.load(f)

    if args.preview_only:
        generate_ir_preview(wflw_dir, all_train_recs, Path(args.preview_path))
        return

    if args.smoke_test:
        run_smoke_test(wflw_dir, base_weights)
        return

    total_epochs = args.stage1_epochs + args.stage2_epochs
    train_recs, val_recs = split_train_val_by_image(all_train_recs, val_ratio=0.10, seed=args.seed)
    print(
        f"[DATA] Train={len(train_recs)} | Val={len(val_recs)} | Test={len(test_recs)} | "
        f"Device={device} | AMP={amp_dtype} | S1=({args.stage1_epochs}ep, bs={args.stage1_batch_size}x{args.stage1_grad_accum}) | "
        f"S2=({args.stage2_epochs}ep, bs={args.stage2_batch_size}x{args.stage2_grad_accum}) | eval_bs={args.eval_batch_size}",
        flush=True,
    )

    db = DatabaseStruct("wflw")
    align_cfg = AlignConfig("wflw", mode="train")

    train_ds = WFLWIRDataset(
        train_recs, wflw_dir, db, align_cfg.posit_ids, is_train=True, ir_prob=0.85, deterministic_ir=False, seed=args.seed
    )
    val_clean_ds = WFLWIRDataset(
        val_recs, wflw_dir, db, align_cfg.posit_ids, is_train=False, ir_prob=0.0
    )
    val_ir_ds = WFLWIRDataset(
        val_recs, wflw_dir, db, align_cfg.posit_ids, is_train=False, ir_prob=1.0, deterministic_ir=True, seed=1000
    )

    sample_weights = compute_sample_weights(train_recs)
    num_workers = args.num_workers
    pin_mem = device.type == "cuda"
    val_clean_loader = make_dataloader(
        val_clean_ds, batch_size=args.eval_batch_size, num_workers=num_workers, pin_memory=pin_mem, shuffle=False
    )
    val_ir_loader = make_dataloader(
        val_ir_ds, batch_size=args.eval_batch_size, num_workers=num_workers, pin_memory=pin_mem, shuffle=False
    )

    model = build_spiga_model(base_weights, device)
    orig_trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}

    pt_weights_np = np.ones((1, 98, 1), dtype=np.float32)
    for idx in range(98):
        if idx in VF50_SET:
            pt_weights_np[0, idx, 0] = 2.0
    pt_weights = torch.from_numpy(pt_weights_np).to(device)

    write_progress(
        progress_path,
        {
            "status": "evaluating_baseline",
            "epoch": 0,
            "total_epochs": total_epochs,
            "amp_dtype": str(amp_dtype),
        },
    )

    # Epoch 0 Baseline evaluation
    base_val_clean = evaluate_dataset(model, val_clean_loader, device, amp_dtype=amp_dtype)
    base_val_ir = evaluate_dataset(model, val_ir_loader, device, amp_dtype=amp_dtype)
    print(
        f"[EPOCH 0 Baseline] Clean Val VF50-NME={base_val_clean['vf50_nme']:.3f}% (WFLW98={base_val_clean['wflw98_nme']:.3f}%) | "
        f"IR Val VF50-NME={base_val_ir['vf50_nme']:.3f}% (WFLW98={base_val_ir['wflw98_nme']:.3f}%)",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_ir_vf50 = base_val_ir["vf50_nme"]
    best_state = clone_state_dict_cpu(model)
    history: List[Dict[str, Any]] = [
        {"stage": 0, "epoch": 0, "val_clean": base_val_clean, "val_ir": base_val_ir}
    ]
    write_progress(
        progress_path,
        {
            "status": "running",
            "epoch": 0,
            "total_epochs": total_epochs,
            "best_ir_vf50_nme": best_ir_vf50,
            "history": history,
        },
    )

    # Stage 1: Fast frozen-stem + hgs[3] + pose/GAT heads fine-tuning
    if args.stage1_epochs > 0:
        set_stage1_freeze(model, orig_trainable_names)
        best_ir_vf50, best_state = run_stage(
            stage_id=1,
            stage_epochs=args.stage1_epochs,
            stage_bs=args.stage1_batch_size,
            stage_grad_accum=args.stage1_grad_accum,
            backbone_lr=args.stage1_backbone_lr,
            head_lr=args.stage1_head_lr,
            epoch_offset=0,
            total_epochs=total_epochs,
            model=model,
            train_ds=train_ds,
            sample_weights=sample_weights,
            val_clean_loader=val_clean_loader,
            val_ir_loader=val_ir_loader,
            device=device,
            amp_dtype=amp_dtype,
            pt_weights=pt_weights,
            num_workers=num_workers,
            log_every=args.log_every,
            out_weights=out_weights,
            progress_path=progress_path,
            best_ir_vf50=best_ir_vf50,
            best_state=best_state,
            history=history,
        )

    # Stage 2: Load best Stage 1 checkpoint and polish full network end-to-end
    if args.stage2_epochs > 0:
        model.load_state_dict(best_state)
        set_stage2_unfreeze(model, orig_trainable_names)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best_ir_vf50, best_state = run_stage(
            stage_id=2,
            stage_epochs=args.stage2_epochs,
            stage_bs=args.stage2_batch_size,
            stage_grad_accum=args.stage2_grad_accum,
            backbone_lr=args.stage2_backbone_lr,
            head_lr=args.stage2_head_lr,
            epoch_offset=args.stage1_epochs,
            total_epochs=total_epochs,
            model=model,
            train_ds=train_ds,
            sample_weights=sample_weights,
            val_clean_loader=val_clean_loader,
            val_ir_loader=val_ir_loader,
            device=device,
            amp_dtype=amp_dtype,
            pt_weights=pt_weights,
            num_workers=num_workers,
            log_every=args.log_every,
            out_weights=out_weights,
            progress_path=progress_path,
            best_ir_vf50=best_ir_vf50,
            best_state=best_state,
            history=history,
        )

    # Release validation persistent workers before spawning test workers (protects 16GB system RAM)
    del val_clean_loader, val_ir_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Ensure best checkpoint is saved even if epoch 0 were never beaten
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(out_weights))

    # Official test.json benchmark (2,500 images): Baseline vs Fine-tuned IR
    print("\n[BENCHMARK] Evaluating Baseline vs Fine-tuned IR on 2,500 test.json images...", flush=True)
    write_progress(
        progress_path,
        {
            "status": "benchmarking",
            "epoch": total_epochs,
            "total_epochs": total_epochs,
            "best_ir_vf50_nme": best_ir_vf50,
            "history": history,
        },
    )
    subsets = db_info.get("test_subsets", {})
    wanted_subsets = {k: subsets[k] for k in ("pose", "blur", "occlusion", "illumination") if k in subsets}

    test_clean_ds = WFLWIRDataset(test_recs, wflw_dir, db, align_cfg.posit_ids, is_train=False, ir_prob=0.0)
    test_ir_ds = WFLWIRDataset(
        test_recs, wflw_dir, db, align_cfg.posit_ids, is_train=False, ir_prob=1.0, deterministic_ir=True, seed=7777
    )
    test_clean_loader = make_dataloader(
        test_clean_ds, batch_size=args.eval_batch_size, num_workers=num_workers, pin_memory=pin_mem, shuffle=False
    )
    test_ir_loader = make_dataloader(
        test_ir_ds, batch_size=args.eval_batch_size, num_workers=num_workers, pin_memory=pin_mem, shuffle=False
    )

    # Evaluate baseline by reloading onto the single existing model (avoids holding 2 models in VRAM)
    base_state = torch.load(str(base_weights), map_location="cpu", weights_only=False)
    model.load_state_dict(base_state)
    del base_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_test_clean = evaluate_dataset(model, test_clean_loader, device, test_recs, wanted_subsets, amp_dtype=amp_dtype)
    print(f"[BENCHMARK] Baseline Clean RGB VF50-NME={base_test_clean['vf50_nme']:.3f}%", flush=True)
    base_test_ir = evaluate_dataset(model, test_ir_loader, device, test_recs, wanted_subsets, amp_dtype=amp_dtype)
    print(f"[BENCHMARK] Baseline Synthetic IR VF50-NME={base_test_ir['vf50_nme']:.3f}%", flush=True)

    model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ft_test_clean = evaluate_dataset(model, test_clean_loader, device, test_recs, wanted_subsets, amp_dtype=amp_dtype)
    print(f"[BENCHMARK] Fine-tuned Clean RGB VF50-NME={ft_test_clean['vf50_nme']:.3f}%", flush=True)
    ft_test_ir = evaluate_dataset(model, test_ir_loader, device, test_recs, wanted_subsets, amp_dtype=amp_dtype)
    print(f"[BENCHMARK] Fine-tuned Synthetic IR VF50-NME={ft_test_ir['vf50_nme']:.3f}%", flush=True)

    del test_clean_loader, test_ir_loader
    gc.collect()

    report = {
        "stage1_epochs": args.stage1_epochs,
        "stage1_batch_size": args.stage1_batch_size,
        "stage1_grad_accum": args.stage1_grad_accum,
        "stage2_epochs": args.stage2_epochs,
        "stage2_batch_size": args.stage2_batch_size,
        "stage2_grad_accum": args.stage2_grad_accum,
        "eval_batch_size": args.eval_batch_size,
        "amp_dtype": str(amp_dtype),
        "train_count": len(train_recs),
        "val_count": len(val_recs),
        "test_count": len(test_recs),
        "history": history,
        "test_baseline_clean": base_test_clean,
        "test_finetuned_clean": ft_test_clean,
        "test_baseline_ir": base_test_ir,
        "test_finetuned_ir": ft_test_ir,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== TEST SET SUMMARY (2,500 images) ===", flush=True)
    print(f"Clean RGB   : Baseline VF50={base_test_clean['vf50_nme']:.3f}% -> Fine-tuned VF50={ft_test_clean['vf50_nme']:.3f}%", flush=True)
    print(f"Synthetic IR: Baseline VF50={base_test_ir['vf50_nme']:.3f}% -> Fine-tuned VF50={ft_test_ir['vf50_nme']:.3f}%", flush=True)
    for sub in ("pose", "blur", "occlusion", "illumination"):
        k = f"{sub}_vf50_nme"
        if k in base_test_ir and k in ft_test_ir:
            print(f"  IR {sub:12s}: {base_test_ir[k]:.3f}% -> {ft_test_ir[k]:.3f}%", flush=True)
    print(f"[DONE] Saved best weights to {out_weights} and report to {report_path}", flush=True)

    write_progress(
        progress_path,
        {
            "status": "completed",
            "epoch": total_epochs,
            "total_epochs": total_epochs,
            "best_ir_vf50_nme": best_ir_vf50,
            "summary": {
                "clean_baseline_vf50": base_test_clean["vf50_nme"],
                "clean_finetuned_vf50": ft_test_clean["vf50_nme"],
                "ir_baseline_vf50": base_test_ir["vf50_nme"],
                "ir_finetuned_vf50": ft_test_ir["vf50_nme"],
            },
            "out_weights": str(out_weights),
            "report_path": str(report_path),
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SPIGA WFLW-98 for Cabin IR VF-50 (2-Stage BF16)")
    p.add_argument("--wflw-dir", default=str(DEFAULT_WFLW_DIR))
    p.add_argument("--base-weights", default=str(DEFAULT_BASE_WEIGHTS))
    p.add_argument("--out-weights", default=str(DEFAULT_OUT_WEIGHTS))
    p.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    p.add_argument("--progress-path", default=str(DEFAULT_PROGRESS_PATH))
    p.add_argument("--stage1-epochs", type=int, default=4)
    p.add_argument("--stage1-batch-size", type=int, default=24)
    p.add_argument("--stage1-grad-accum", type=int, default=2)
    p.add_argument("--stage1-backbone-lr", type=float, default=1e-4)
    p.add_argument("--stage1-head-lr", type=float, default=2e-4)
    p.add_argument("--stage2-epochs", type=int, default=2)
    p.add_argument("--stage2-batch-size", type=int, default=14)
    p.add_argument("--stage2-grad-accum", type=int, default=2)
    p.add_argument("--stage2-backbone-lr", type=float, default=2.5e-5)
    p.add_argument("--stage2-head-lr", type=float, default=6e-5)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--preview-only", action="store_true")
    p.add_argument(
        "--preview-path",
        default=r"C:\Users\ADMIN\.gemini\antigravity\brain\da258e0a-ab97-4e17-a3e6-c1b4db434671\scratch\ir_aug_preview.png",
    )
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
