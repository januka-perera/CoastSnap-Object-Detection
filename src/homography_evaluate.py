"""GCP-based evaluation for deep homography estimation.

For each image with a .mat file, predicts the homography H (target -> reference),
transforms the target GCPs by H, and compares with reference GCPs. Reports
Euclidean pixel error statistics.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import yaml

from .homography_predict import load_homography_model, predict_homography
from .homography_utils import transform_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_gcps_from_mat(mat_path: str, image_path: str) -> np.ndarray:
    """Load GCP pixel coordinates from a .mat file.

    Scales from annotation resolution (NU x NV) to actual image resolution.

    Args:
        mat_path: Path to .mat file.
        image_path: Path to corresponding image (for resolution).

    Returns:
        (N, 2) array of (u, v) pixel coordinates.
    """
    data = scipy.io.loadmat(str(mat_path))
    uv = data["metadata"]["gcps"][0, 0]["UVpicked"][0, 0]  # (N, 2)
    lcp = data["metadata"]["geom"][0, 0]["lcp"][0, 0]
    nu = lcp["NU"][0, 0].item()
    nv = lcp["NV"][0, 0].item()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_h, img_w = img.shape[:2]

    scale_u = img_w / nu
    scale_v = img_h / nv

    coords = np.zeros_like(uv, dtype=np.float64)
    coords[:, 0] = uv[:, 0] * scale_u
    coords[:, 1] = uv[:, 1] * scale_v
    return coords


def evaluate_homography(config_path: str = "configs/config.yaml", checkpoint_path: str = None):
    """Evaluate homography estimation using GCP reprojection error.

    For each image with a .mat GCP file:
    1. Predict H (target -> reference)
    2. Transform target GCPs by H
    3. Compare with reference GCPs
    4. Report Euclidean pixel error

    Args:
        config_path: Path to config yaml.
        checkpoint_path: Path to model checkpoint.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hcfg = cfg["homography"]

    # Load model
    model, reference, mask, cfg, device, ref_full_size = load_homography_model(config_path, checkpoint_path)

    # Load reference GCPs
    ref_image_path = hcfg["reference_image"]
    ref_mat_path = hcfg.get("reference_mat",
                             cfg.get("refinement", {}).get("reference_mat"))
    if ref_mat_path is None:
        raise ValueError("reference_mat not found in config (homography or refinement section)")

    ref_gcps = load_gcps_from_mat(ref_mat_path, ref_image_path)
    num_gcps = ref_gcps.shape[0]
    logger.info(f"Reference GCPs: {num_gcps} points")

    # Find all images with .mat files
    images_dir = Path(cfg["data"]["images_dir"])
    gcp_dir = Path("data/gcp")
    ref_name = Path(ref_image_path).name

    all_errors = []
    per_gcp_errors = {i: [] for i in range(num_gcps)}
    results = []

    mat_files = sorted(gcp_dir.glob("*.mat"))
    logger.info(f"Found {len(mat_files)} .mat files")

    for mat_path in mat_files:
        # Find corresponding image
        img_name = mat_path.name.replace(".plan.", ".snap.").replace(".mat", ".jpg")
        img_path = images_dir / img_name

        if not img_path.exists():
            continue

        # Skip reference image
        if img_name == ref_name:
            continue

        try:
            # Load target GCPs
            target_gcps = load_gcps_from_mat(str(mat_path), str(img_path))
            if target_gcps.shape[0] != num_gcps:
                logger.warning(f"Skipping {img_name}: {target_gcps.shape[0]} GCPs != {num_gcps}")
                continue

            # Predict homography (with iterative refinement)
            num_iters = cfg["homography"].get("num_iterations", 1)
            H_full, four_point = predict_homography(
                str(img_path), model, reference, cfg, device,
                num_iterations=num_iters,
                ref_full_size=ref_full_size,
            )

            # Transform target GCPs by H
            transformed_gcps = transform_points(target_gcps, H_full)

            # Compute per-GCP errors
            errors = np.sqrt(np.sum((transformed_gcps - ref_gcps) ** 2, axis=1))

            sample_result = {
                "filename": img_name,
                "mean_error_px": round(float(np.mean(errors)), 1),
                "max_error_px": round(float(np.max(errors)), 1),
                "per_gcp": [],
            }

            for i in range(num_gcps):
                all_errors.append(errors[i])
                per_gcp_errors[i].append(errors[i])
                sample_result["per_gcp"].append({
                    "id": i,
                    "error_px": round(float(errors[i]), 1),
                    "target_uv": [round(float(target_gcps[i, 0]), 1),
                                  round(float(target_gcps[i, 1]), 1)],
                    "transformed_uv": [round(float(transformed_gcps[i, 0]), 1),
                                       round(float(transformed_gcps[i, 1]), 1)],
                    "reference_uv": [round(float(ref_gcps[i, 0]), 1),
                                     round(float(ref_gcps[i, 1]), 1)],
                })

            results.append(sample_result)
            err_level = "WARNING" if np.mean(errors) > 200 else "INFO"
            getattr(logger, err_level.lower())(
                f"{img_name}: mean={np.mean(errors):.1f}px, max={np.max(errors):.1f}px"
            )

        except Exception as e:
            logger.error(f"Failed on {img_name}: {e}")

    if not all_errors:
        logger.warning("No images evaluated!")
        return {}

    all_errors = np.array(all_errors)

    # Compute summary metrics
    metrics = {
        "num_images": len(results),
        "num_gcps_total": len(all_errors),
        "mean_error_px": round(float(np.mean(all_errors)), 1),
        "median_error_px": round(float(np.median(all_errors)), 1),
        "p90_error_px": round(float(np.percentile(all_errors, 90)), 1),
        "std_error_px": round(float(np.std(all_errors)), 1),
        "max_error_px": round(float(np.max(all_errors)), 1),
    }

    for threshold in [5, 10, 20, 50]:
        pck = float(np.mean(all_errors < threshold) * 100)
        metrics[f"pck@{threshold}px"] = round(pck, 1)

    metrics["per_gcp"] = {}
    for gid, errs in per_gcp_errors.items():
        if errs:
            errs = np.array(errs)
            metrics["per_gcp"][gid] = {
                "mean_error_px": round(float(np.mean(errs)), 1),
                "median_error_px": round(float(np.median(errs)), 1),
                "count": len(errs),
            }

    # Save results
    results_dir = Path(cfg["output"]["results_dir"]) / "homography"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("HOMOGRAPHY EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Images evaluated: {metrics['num_images']}")
    logger.info(f"  Mean error:       {metrics['mean_error_px']:.1f} px")
    logger.info(f"  Median error:     {metrics['median_error_px']:.1f} px")
    logger.info(f"  90th percentile:  {metrics['p90_error_px']:.1f} px")
    logger.info(f"  Max error:        {metrics['max_error_px']:.1f} px")
    for t in [5, 10, 20, 50]:
        logger.info(f"  PCK@{t}px:         {metrics[f'pck@{t}px']:.1f}%")
    logger.info("-" * 60)
    logger.info("Per-GCP breakdown:")
    for gid, stats in metrics["per_gcp"].items():
        logger.info(f"  GCP {gid}: mean={stats['mean_error_px']:.1f}px, "
                     f"median={stats['median_error_px']:.1f}px (n={stats['count']})")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    evaluate_homography(args.config, args.checkpoint)
