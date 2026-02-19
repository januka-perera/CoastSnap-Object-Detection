"""
Evaluation for detection and keypoint models.

Modes
-----
detection  Run YOLO validation → mAP@50, mAP@50-95, precision, recall.
keypoint   Evaluate heatmap model on pre-cropped test images → EPE, RMSE, PCK.
pipeline   Run both.

Usage
-----
    # Detection only
    python evaluate.py --mode detection \
                       --yolo-weights ./yolo_runs/phase3_full/weights/best.pt

    # Keypoint only
    python evaluate.py --mode keypoint \
                       --kp-weights ./keypoint_checkpoints/keypoint_best.pt

    # Full pipeline
    python evaluate.py --mode pipeline \
                       --yolo-weights ./yolo_runs/phase3_full/weights/best.pt \
                       --kp-weights   ./keypoint_checkpoints/keypoint_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model         import KeypointHeatmapModel
from heatmap_utils import extract_coordinate


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Detection evaluation (delegates to ultralytics)
# ---------------------------------------------------------------------------

def evaluate_yolo(
    weights: str,
    data_yaml: str,
    cfg: dict,
    split: str = "test",
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        sys.exit(1)

    model   = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=cfg["yolo"]["imgsz"],
        conf=cfg["yolo"]["conf_threshold"],
        iou=cfg["yolo"]["iou_threshold"],
        verbose=True,
    )

    results = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }

    print(f"\nDetection ({split}):")
    print(f"  mAP@50     : {results['mAP50']:.4f}")
    print(f"  mAP@50-95  : {results['mAP50_95']:.4f}")
    print(f"  Precision  : {results['precision']:.4f}")
    print(f"  Recall     : {results['recall']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Keypoint evaluation
# ---------------------------------------------------------------------------

def evaluate_keypoint(
    weights: str,
    cfg: dict,
    split: str = "test",
    subpixel: bool = True,
) -> dict:
    """
    Evaluate the keypoint model on pre-cropped images.

    Returns a dict with:
        n_samples, epe_mean, epe_std, median_epe, rmse, pck_5, pck_10
    where EPE is measured in heatmap pixels.
    """
    from dataset       import KeypointCropDataset
    from torch.utils.data import DataLoader

    kp_cfg   = cfg["keypoint"]
    data_cfg = cfg["data"]

    crops_dir    = Path(data_cfg["crops_dir"]) / split
    input_size   = tuple(kp_cfg["input_size"])
    heatmap_size = tuple(kp_cfg["heatmap_size"])
    sigma        = kp_cfg["sigma"]

    dataset = KeypointCropDataset(crops_dir, input_size, heatmap_size, sigma, augment=False)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = KeypointHeatmapModel(
        pretrained=False,
        decoder_channels=kp_cfg.get("decoder_channels", [128, 64, 32]),
    )
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval().to(device)

    hm_h, hm_w = heatmap_size
    all_epes    = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt_kp  = batch["keypoint"]        # (B, 2) normalised, CPU

            pred_hm = model(images)            # (B, 1, hm_h, hm_w)
            pred_np = pred_hm.squeeze(1).cpu().numpy()

            for i in range(pred_np.shape[0]):
                xp, yp = extract_coordinate(pred_np[i], hm_h, hm_w, subpixel=subpixel)
                xg     = gt_kp[i, 0].item()
                yg     = gt_kp[i, 1].item()
                epe    = np.hypot(
                    (xp - xg) * hm_w,
                    (yp - yg) * hm_h,
                )
                all_epes.append(epe)

    epes = np.array(all_epes)
    results = {
        "n_samples":  int(len(epes)),
        "epe_mean":   float(epes.mean()),
        "epe_std":    float(epes.std()),
        "median_epe": float(np.median(epes)),
        "rmse":       float(np.sqrt((epes ** 2).mean())),
        "pck_5":      float((epes < 5.0).mean()),
        "pck_10":     float((epes < 10.0).mean()),
    }

    print(f"\nKeypoint evaluation ({split}):")
    print(f"  Samples    : {results['n_samples']}")
    print(f"  EPE mean   : {results['epe_mean']:.2f} px")
    print(f"  EPE median : {results['median_epe']:.2f} px")
    print(f"  EPE std    : {results['epe_std']:.2f} px")
    print(f"  RMSE       : {results['rmse']:.2f} px")
    print(f"  PCK @ 5px  : {results['pck_5'] * 100:.1f} %")
    print(f"  PCK @ 10px : {results['pck_10'] * 100:.1f} %")

    return results, epes.tolist()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_epe_distribution(epes: list, output_path: str):
    arr  = np.array(epes)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr, bins=30, edgecolor="black", alpha=0.75, color="steelblue")
    ax.axvline(arr.mean(),   color="red",    linestyle="--",
               label=f"Mean   = {arr.mean():.1f} px")
    ax.axvline(np.median(arr), color="orange", linestyle="--",
               label=f"Median = {np.median(arr):.1f} px")
    ax.set_xlabel("EPE (heatmap pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Keypoint Prediction Error Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved EPE plot: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate detection / keypoint models")
    parser.add_argument("--config",      default="../configs/config.yaml")
    parser.add_argument("--mode",        default="keypoint",
                        choices=["detection", "keypoint", "pipeline"])
    parser.add_argument("--yolo-weights",default=None)
    parser.add_argument("--kp-weights",  default=None)
    parser.add_argument("--split",       default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output-dir",  default="./eval_results")
    parser.add_argument("--no-subpixel", action="store_true")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── Detection ─────────────────────────────────────────────────────────
    if args.mode in ("detection", "pipeline"):
        if not args.yolo_weights:
            print("ERROR: --yolo-weights required for detection/pipeline mode.")
            sys.exit(1)
        data_yaml = str(Path(cfg["data"]["yolo_data_dir"]) / "dataset.yaml")
        det_res   = evaluate_yolo(args.yolo_weights, data_yaml, cfg, split=args.split)
        all_results["detection"] = det_res

    # ── Keypoint ──────────────────────────────────────────────────────────
    if args.mode in ("keypoint", "pipeline"):
        if not args.kp_weights:
            print("ERROR: --kp-weights required for keypoint/pipeline mode.")
            sys.exit(1)
        kp_res, epes = evaluate_keypoint(
            args.kp_weights, cfg,
            split=args.split,
            subpixel=not args.no_subpixel,
        )
        all_results["keypoint"] = kp_res
        plot_epe_distribution(epes, str(out_dir / f"epe_dist_{args.split}.png"))

    # Save JSON summary
    summary_path = out_dir / f"eval_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
