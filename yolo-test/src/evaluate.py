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
# Detection visualisation
# ---------------------------------------------------------------------------

def visualise_detections(
    weights: str,
    cfg: dict,
    split: str,
    out_dir: Path,
):
    """
    Run YOLO predict on every image in the split and save annotated images.

    Draws predicted boxes (green) and ground-truth boxes (red) side by side
    so misses and false positives are immediately visible.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        return

    import cv2

    yolo_cfg    = cfg["yolo"]
    data_cfg    = cfg["data"]
    class_names = data_cfg.get("classes", [])

    img_dir = Path(data_cfg["yolo_data_dir"]) / "images" / split
    lbl_dir = Path(data_cfg["yolo_data_dir"]) / "labels" / split
    vis_dir = out_dir / f"detections_{split}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model      = YOLO(weights)
    img_paths  = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    print(f"\nVisualising {len(img_paths)} {split} images → {vis_dir}")

    for img_path in img_paths:
        img   = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W  = img.shape[:2]
        canvas = img.copy()

        # ── Ground-truth boxes (red) ──────────────────────────────────────
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_idx = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * W);  y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W);  y2 = int((cy + bh / 2) * H)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 200), 2)
                name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                cv2.putText(canvas, f"GT:{name}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)

        # ── Predicted boxes (green) ───────────────────────────────────────
        results = model.predict(
            str(img_path),
            conf=yolo_cfg["conf_threshold"],
            iou=yolo_cfg["iou_threshold"],
            verbose=False,
        )
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf    = float(box.conf.item())
                cls_idx = int(box.cls.item())
                name    = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(canvas, f"{name} {conf:.2f}", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

        out_path = vis_dir / img_path.name
        cv2.imwrite(str(out_path), canvas)

    print(f"Saved {len(img_paths)} images.  Red = GT, Green = Predicted.")


# ---------------------------------------------------------------------------
# Keypoint visualisation
# ---------------------------------------------------------------------------

def visualise_keypoints(
    kp_weights: str,
    cfg: dict,
    split: str,
    out_dir: Path,
    subpixel: bool = True,
):
    """
    For every crop in the split:
      - Predict the keypoint heatmap
      - Draw GT point (red cross) and predicted point (green dot) on the crop
      - Save the heatmap as a colour overlay beside the crop

    Output layout per image  (side by side):
        [ crop with points ]  |  [ heatmap colourmap ]
    """
    import cv2

    kp_cfg   = cfg["keypoint"]
    data_cfg = cfg["data"]

    crops_dir    = Path(data_cfg["crops_dir"]) / split
    input_size   = tuple(kp_cfg["input_size"])
    heatmap_size = tuple(kp_cfg["heatmap_size"])
    sigma        = kp_cfg["sigma"]

    vis_dir = out_dir / f"keypoints_{split}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = KeypointHeatmapModel(
        pretrained=False,
        decoder_channels=kp_cfg.get("decoder_channels", [128, 64, 32]),
    )
    model.load_state_dict(torch.load(kp_weights, map_location=device))
    model.eval().to(device)

    hm_h, hm_w   = heatmap_size
    in_h, in_w   = input_size
    img_paths    = sorted((crops_dir / "images").glob("*.jpg"))
    lbl_dir      = crops_dir / "labels"

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print(f"\nVisualising {len(img_paths)} keypoint crops → {vis_dir}")

    for img_path in img_paths:
        # ── Load crop ────────────────────────────────────────────────────
        crop_bgr = cv2.imread(str(img_path))
        if crop_bgr is None:
            continue
        crop_h, crop_w = crop_bgr.shape[:2]

        # ── Ground-truth keypoint ─────────────────────────────────────────
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        gt_x_norm, gt_y_norm = 0.5, 0.5
        if lbl_path.exists():
            parts = lbl_path.read_text().strip().split()
            gt_x_norm, gt_y_norm = float(parts[0]), float(parts[1])

        # ── Predict heatmap ───────────────────────────────────────────────
        inp = cv2.resize(crop_bgr, (in_w, in_h))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (inp - _MEAN) / _STD
        t   = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            hm = model(t)
        hm_np = hm.squeeze().cpu().numpy()          # (hm_h, hm_w)

        # ── Extract predicted coordinate ──────────────────────────────────
        pred_x_norm, pred_y_norm = extract_coordinate(
            hm_np, hm_h, hm_w, subpixel=subpixel
        )

        # ── Draw on a resized version of the crop ─────────────────────────
        canvas = cv2.resize(crop_bgr, (in_w, in_h))

        # GT point — red cross
        gx = int(gt_x_norm   * in_w)
        gy = int(gt_y_norm   * in_h)
        cv2.drawMarker(canvas, (gx, gy), (0, 0, 220),
                       cv2.MARKER_CROSS, markerSize=18, thickness=2)

        # Predicted point — green filled circle
        px = int(pred_x_norm * in_w)
        py = int(pred_y_norm * in_h)
        cv2.circle(canvas, (px, py), 6,  (0, 210, 0), -1)
        cv2.circle(canvas, (px, py), 8,  (255, 255, 255), 1)

        # EPE in heatmap pixels
        epe = np.hypot((pred_x_norm - gt_x_norm) * hm_w,
                       (pred_y_norm - gt_y_norm) * hm_h)
        cv2.putText(canvas, f"EPE {epe:.1f}px", (4, in_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Legend
        cv2.putText(canvas, "GT",   (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
        cv2.putText(canvas, "Pred", (4, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 0), 1)

        # ── Heatmap colourmap panel ────────────────────────────────────────
        hm_norm  = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-8)
        hm_uint8 = (hm_norm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_color = cv2.resize(hm_color, (in_w, in_h))

        # Mark predicted peak on heatmap too
        cv2.circle(hm_color, (px, py), 6,  (255, 255, 255), -1)
        cv2.circle(hm_color, (px, py), 8,  (0, 0, 0), 1)

        # ── Combine side by side and save ─────────────────────────────────
        combined = np.concatenate([canvas, hm_color], axis=1)
        cv2.imwrite(str(vis_dir / img_path.name), combined)

    print(f"Saved {len(img_paths)} images.  Red cross = GT, Green dot = Predicted.")


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
    _default_cfg = str(Path(__file__).parent.parent / "configs" / "config.yaml")
    parser.add_argument("--config",      default=_default_cfg)
    parser.add_argument("--mode",        default="keypoint",
                        choices=["detection", "keypoint", "pipeline"])
    parser.add_argument("--yolo-weights",default=None)
    parser.add_argument("--kp-weights",  default=None)
    parser.add_argument("--split",       default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output-dir",  default="./eval_results")
    parser.add_argument("--no-subpixel",   action="store_true")
    parser.add_argument("--save-images",   action="store_true",
                        help="Save annotated detection images (GT=red, Pred=green).")
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

        if args.save_images:
            visualise_detections(args.yolo_weights, cfg, args.split, out_dir)

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

        if args.save_images:
            visualise_keypoints(
                args.kp_weights, cfg, args.split, out_dir,
                subpixel=not args.no_subpixel,
            )

    # Save JSON summary
    summary_path = out_dir / f"eval_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
