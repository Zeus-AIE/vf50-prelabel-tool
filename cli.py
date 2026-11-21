"""CLI entry point for batch VF-50 facial landmark pre-labeling."""

from __future__ import annotations

import argparse
import os
import time

from src.detector import VALID_DETECTORS, VALID_DRIVER_SIDES, VALID_ROTATIONS
from src.pipeline import VALID_ENSEMBLES, VALID_HEADPOSES, VF50Pipeline, get_level_presets


def build_parser() -> argparse.ArgumentParser:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_models = os.path.join(root_dir, "models")

    p = argparse.ArgumentParser(
        description="Local Pre-labeling Tool cho Facial Landmark-50 (VF-50) xuat chuan CVAT 1.1"
    )
    p.add_argument("--input", required=True, help="Duong dan 1 anh hoac thu muc anh dau vao")
    p.add_argument("--output", required=True, help="Thu muc xuat annotations.xml, qc_report.csv, previews")
    p.add_argument(
        "--rotate",
        default="0",
        choices=VALID_ROTATIONS,
        help="Xoay anh truoc khi detect (0, 90cw, 90ccw, 180, auto). Mac dinh: 0",
    )
    p.add_argument(
        "--driver-side",
        default="largest",
        choices=VALID_DRIVER_SIDES,
        help="Cach chon khuon mat nguoi lai khi co nhieu mat (largest, left, right, center)",
    )
    p.add_argument(
        "--yaw-occluded",
        type=float,
        default=35.0,
        help="Nguong |yaw| (do) de bat dau gan Occluded cho nua mat phia xa",
    )
    p.add_argument(
        "--yaw-outside",
        type=float,
        default=55.0,
        help="Nguong |yaw| (do) de gan Outside=1 toan bo mat/long may phia xa",
    )
    p.add_argument(
        "--template-xml",
        default=None,
        help="Duong dan file annotations.xml export tu CVAT task de giu nguyen <meta> va <image>",
    )
    p.add_argument(
        "--save-previews",
        action="store_true",
        help="Luu anh preview ve 50 keypoints + trang thai vao output/previews/",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Thiet bi chay SPIGA (cuda hoac cpu)",
    )
    p.add_argument(
        "--models-dir",
        default=default_models,
        help="Thu muc chua weights SPIGA (*.pt) va face_detection_yunet_2023mar.onnx",
    )
    p.add_argument(
        "--level",
        type=int,
        default=None,
        choices=(1, 2, 3),
        help="Chon nhanh muc do 1 (Nhanh), 2 (Pretrained), 3 (IR Fine-tuned)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Ten file weights SPIGA trong models-dir (mac dinh uu tien spiga_wflw_ir_vf50.pt neu co)",
    )
    p.add_argument(
        "--detector",
        default=None,
        choices=VALID_DETECTORS,
        help="Model detect khuon mat (scrfd10g hoac yunet). Mac dinh: scrfd10g",
    )
    p.add_argument(
        "--ensemble",
        default=None,
        choices=VALID_ENSEMBLES,
        help="To hop landmark (spiga_tufa hoac spiga_only). Mac dinh: spiga_tufa",
    )
    p.add_argument(
        "--headpose",
        default=None,
        choices=VALID_HEADPOSES,
        help="Model uoc luong goc dau (6drepnet hoac spiga). Mac dinh: 6drepnet",
    )
    p.add_argument(
        "--no-tta",
        action="store_true",
        help="Tat Horizontal Flip TTA khi suy luan landmark & headpose",
    )
    p.add_argument(
        "--no-two-pass",
        action="store_true",
        help="Tat 2-Pass coarse-to-fine bbox refinement",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    t0 = time.time()

    weights_name = args.weights
    detector_mode = "scrfd10g"
    ensemble_mode = "spiga_tufa"
    headpose_mode = "6drepnet"
    use_tta = True
    use_two_pass = True

    if args.level is not None:
        presets = get_level_presets(args.models_dir)["level_presets"]
        preset = next(p for p in presets if p["id"] == f"level{args.level}")
        cfg = preset["config"]
        weights_name = weights_name or cfg["weights_name"]
        detector_mode = cfg["detector_mode"]
        ensemble_mode = cfg["ensemble_mode"]
        headpose_mode = cfg["headpose_mode"]
        use_tta = bool(cfg["use_tta"])
        use_two_pass = bool(cfg["use_two_pass"])

    if args.detector is not None:
        detector_mode = args.detector
    if args.ensemble is not None:
        ensemble_mode = args.ensemble
    if args.headpose is not None:
        headpose_mode = args.headpose
    if args.no_tta:
        use_tta = False
    if args.no_two_pass:
        use_two_pass = False

    print(f"[VF-50] Dang khoi tao model tu {args.models_dir} (device={args.device})...")
    pipeline = VF50Pipeline(
        models_dir=args.models_dir,
        device=args.device,
        weights_name=weights_name,
        detector_mode=detector_mode,
        ensemble_mode=ensemble_mode,
        headpose_mode=headpose_mode,
        use_tta=use_tta,
        use_two_pass=use_two_pass,
    )
    print(
        f"[VF-50] Stack: weights={pipeline.weights_name} | detector={pipeline.default_detector_mode} | "
        f"ensemble={pipeline.default_ensemble_mode} | headpose={pipeline.default_headpose_mode} | "
        f"TTA={pipeline.use_tta} | 2-Pass={pipeline.use_two_pass}"
    )

    def _on_progress(idx: int, total: int, name: str) -> None:
        print(f"[{idx}/{total}] Da xu ly: {name}")

    preds = pipeline.run_batch(
        input_path=args.input,
        output_dir=args.output,
        rotate_mode=args.rotate,
        driver_side=args.driver_side,
        yaw_occluded=args.yaw_occluded,
        yaw_outside=args.yaw_outside,
        template_xml_path=args.template_xml,
        save_previews=args.save_previews,
        progress_cb=_on_progress,
    )

    detected_cnt = sum(1 for p in preds if p.detected)
    flagged_cnt = sum(1 for p in preds if p.qc_flags)
    elapsed = time.time() - t0
    print(
        f"\n[HOAN TAT] Tong: {len(preds)} anh | Detect duoc: {detected_cnt}/{len(preds)} | "
        f"Can QC review: {flagged_cnt} | Thoi gian: {elapsed:.2f}s"
    )
    print(f"  - CVAT XML : {os.path.join(args.output, 'annotations.xml')}")
    print(f"  - QC CSV   : {os.path.join(args.output, 'qc_report.csv')}")
    print(f"  - JSON     : {os.path.join(args.output, 'predictions.json')}")
    if args.save_previews:
        print(f"  - Previews : {os.path.join(args.output, 'previews')}")


if __name__ == "__main__":
    main()
