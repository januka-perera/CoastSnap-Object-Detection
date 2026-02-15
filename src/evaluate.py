"""Evaluation and metrics for heatmap regression model."""

import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from .dataset import LandmarkDataset
from .model import HeatmapRegressor
from .heatmap_utils import soft_argmax_coordinates, argmax_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(config_path: str = "configs/config.yaml", checkpoint_path: str = None):
    """Evaluate model on test set.

    Args:
        config_path: Path to config yaml.
        checkpoint_path: Path to model checkpoint. If None, uses best_model.pth.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if checkpoint_path is None:
        checkpoint_path = Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth"

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HeatmapRegressor(
        num_landmarks=cfg["data"]["num_landmarks"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test split
    log_dir = Path(cfg["output"]["log_dir"])
    with open(log_dir / "data_split.json", "r") as f:
        split = json.load(f)
    test_indices = split["test"]

    # Build dataset (no augmentation)
    dataset = LandmarkDataset(
        annotations_file=cfg["data"]["annotations_file"],
        images_dir=cfg["data"]["images_dir"],
        num_landmarks=cfg["data"]["num_landmarks"],
        input_size=(cfg["data"]["input_width"], cfg["data"]["input_height"]),
        output_size=(cfg["data"]["output_width"], cfg["data"]["output_height"]),
        sigma=cfg["heatmap"]["sigma"],
        augment=False,
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    use_soft = cfg["inference"]["soft_argmax"]
    window_size = cfg["inference"]["local_window_size"]

    all_errors = []           # per-landmark errors in pixels
    per_landmark_errors = {}  # landmark_id -> list of errors
    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            visibility = batch["visibility"]
            keypoints_gt = batch["keypoints"]
            orig_w = batch["orig_width"].item()
            orig_h = batch["orig_height"].item()
            filename = batch["filename"][0]

            heatmaps_pred = model(images)

            if use_soft:
                pred_coords = soft_argmax_coordinates(heatmaps_pred, window_size)
            else:
                pred_coords = argmax_coordinates(heatmaps_pred)

            pred_coords = pred_coords.cpu()

            # Compute per-landmark errors
            N = cfg["data"]["num_landmarks"]
            sample_result = {"filename": filename, "landmarks": []}

            for n in range(N):
                if not visibility[0, n]:
                    continue

                pred_u = pred_coords[0, n, 0].item() * orig_w
                pred_v = pred_coords[0, n, 1].item() * orig_h
                gt_u = keypoints_gt[0, n, 0].item() * orig_w
                gt_v = keypoints_gt[0, n, 1].item() * orig_h

                error = np.sqrt((pred_u - gt_u) ** 2 + (pred_v - gt_v) ** 2)
                all_errors.append(error)

                if n not in per_landmark_errors:
                    per_landmark_errors[n] = []
                per_landmark_errors[n].append(error)

                sample_result["landmarks"].append({
                    "id": n,
                    "pred_u": round(pred_u, 1),
                    "pred_v": round(pred_v, 1),
                    "gt_u": round(gt_u, 1),
                    "gt_v": round(gt_v, 1),
                    "error_px": round(error, 1),
                })

            results.append(sample_result)

    # Compute metrics
    all_errors = np.array(all_errors)
    metrics = {
        "mean_error_px": float(np.mean(all_errors)),
        "median_error_px": float(np.median(all_errors)),
        "p90_error_px": float(np.percentile(all_errors, 90)),
        "std_error_px": float(np.std(all_errors)),
        "num_samples": len(test_indices),
        "num_keypoints_evaluated": len(all_errors),
    }

    # PCK at various thresholds
    for threshold in [5, 10, 20, 50]:
        pck = float(np.mean(all_errors < threshold) * 100)
        metrics[f"pck@{threshold}px"] = round(pck, 1)

    # Per-landmark breakdown
    metrics["per_landmark"] = {}
    for lid, errors in sorted(per_landmark_errors.items()):
        errors = np.array(errors)
        metrics["per_landmark"][lid] = {
            "mean_error_px": round(float(np.mean(errors)), 1),
            "median_error_px": round(float(np.median(errors)), 1),
            "count": len(errors),
        }

    # Save results
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Mean error:   {metrics['mean_error_px']:.1f} px")
    logger.info(f"  Median error: {metrics['median_error_px']:.1f} px")
    logger.info(f"  90th %ile:    {metrics['p90_error_px']:.1f} px")
    for t in [5, 10, 20, 50]:
        logger.info(f"  PCK@{t}px:     {metrics[f'pck@{t}px']:.1f}%")
    logger.info("-" * 60)
    logger.info("Per-landmark breakdown:")
    for lid, stats in metrics["per_landmark"].items():
        logger.info(f"  Landmark {lid}: mean={stats['mean_error_px']:.1f}px, "
                     f"median={stats['median_error_px']:.1f}px (n={stats['count']})")
    logger.info("=" * 60)

    return metrics


def visualise_predictions(
    config_path: str = "configs/config.yaml",
    checkpoint_path: str = None,
    num_images: int = 5,
    save_dir: str = None,
):
    """Visualise predictions overlaid on test images with heatmaps."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HeatmapRegressor(
        num_landmarks=cfg["data"]["num_landmarks"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log_dir = Path(cfg["output"]["log_dir"])
    with open(log_dir / "data_split.json", "r") as f:
        split = json.load(f)
    test_indices = split["test"][:num_images]

    dataset = LandmarkDataset(
        annotations_file=cfg["data"]["annotations_file"],
        images_dir=cfg["data"]["images_dir"],
        num_landmarks=cfg["data"]["num_landmarks"],
        input_size=(cfg["data"]["input_width"], cfg["data"]["input_height"]),
        output_size=(cfg["data"]["output_width"], cfg["data"]["output_height"]),
        sigma=cfg["heatmap"]["sigma"],
        augment=False,
    )

    if save_dir is None:
        save_dir = Path(cfg["output"]["results_dir"]) / "visualisations"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_soft = cfg["inference"]["soft_argmax"]
    window_size = cfg["inference"]["local_window_size"]
    N = cfg["data"]["num_landmarks"]
    colours = plt.cm.tab10(np.linspace(0, 1, N))

    for idx in test_indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        visibility = sample["visibility"]
        keypoints_gt = sample["keypoints"]
        filename = sample["filename"]

        with torch.no_grad():
            heatmaps_pred = model(image)
            if use_soft:
                pred_coords = soft_argmax_coordinates(heatmaps_pred, window_size)
            else:
                pred_coords = argmax_coordinates(heatmaps_pred)

        pred_coords = pred_coords.cpu().squeeze(0)
        heatmaps_np = heatmaps_pred.cpu().squeeze(0).numpy()

        # Load original image for display
        img_path = Path(cfg["data"]["images_dir"]) / filename
        disp_img = cv2.imread(str(img_path))
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = disp_img.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: image with predicted (circle) and GT (x) coordinates
        axes[0].imshow(disp_img)
        axes[0].set_title(f"Predictions: {filename}")
        for n in range(N):
            if not visibility[n]:
                continue
            pu = pred_coords[n, 0].item() * orig_w
            pv = pred_coords[n, 1].item() * orig_h
            gu = keypoints_gt[n, 0].item() * orig_w
            gv = keypoints_gt[n, 1].item() * orig_h

            c = colours[n]
            axes[0].plot(pu, pv, "o", color=c, markersize=8, label=f"Pred {n}")
            axes[0].plot(gu, gv, "x", color=c, markersize=10, markeredgewidth=2)
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].axis("off")

        # Right: composite heatmap
        composite = np.zeros((heatmaps_np.shape[1], heatmaps_np.shape[2], 3))
        for n in range(N):
            if not visibility[n]:
                continue
            for c_idx in range(3):
                composite[:, :, c_idx] += heatmaps_np[n] * colours[n][c_idx]
        composite = np.clip(composite, 0, 1)
        axes[1].imshow(composite)
        axes[1].set_title("Predicted Heatmaps")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_dir / f"{Path(filename).stem}_pred.png", dpi=150)
        plt.close()

    logger.info(f"Saved {len(test_indices)} visualisations to {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--num-vis", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint)
    if args.visualise:
        visualise_predictions(args.config, args.checkpoint, args.num_vis)
