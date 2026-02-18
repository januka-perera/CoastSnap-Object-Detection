"""Inference for deep homography estimation.

Loads a trained HomographyNet, predicts the homography between a target
image and the reference, and warps the target to the reference frame.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .homography_model import HomographyNet
from .homography_utils import (
    four_point_to_homography,
    scale_homography,
    transform_points,
    warp_image,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_homography_model(
    config_path: str = "configs/config.yaml",
    checkpoint_path: str = None,
) -> tuple:
    """Load HomographyNet, reference image, and mask.

    Args:
        config_path: Path to config yaml.
        checkpoint_path: Path to model checkpoint. If None, uses best_homography.pth.

    Returns:
        (model, reference_gray, mask, cfg, device) where reference_gray is
        the grayscale reference at working resolution, and mask is the
        static region mask at working resolution.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hcfg = cfg["homography"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = Path(hcfg["output"]["checkpoint_dir"]) / "best_homography.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = hcfg["model"]
    model = HomographyNet(
        backbone=model_cfg["backbone"],
        pretrained=False,
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load reference at working resolution
    working_w, working_h = hcfg["working_width"], hcfg["working_height"]
    ref_bgr = cv2.imread(str(hcfg["reference_image"]))
    if ref_bgr is None:
        raise FileNotFoundError(f"Reference not found: {hcfg['reference_image']}")
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_working = cv2.resize(ref_gray, (working_w, working_h)).astype(np.float32) / 255.0

    # Load mask
    mask_path = hcfg.get("mask_path")
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (working_w, working_h)).astype(np.float32) / 255.0
    else:
        mask = np.ones((working_h, working_w), dtype=np.float32)

    return model, ref_working, mask, cfg, device


def predict_homography(
    image_path: str,
    model: torch.nn.Module,
    reference: np.ndarray,
    cfg: dict,
    device: torch.device,
    num_iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the homography aligning a target image to the reference.

    Supports iterative refinement: predict H, warp, predict residual,
    compose. Each iteration corrects the residual misalignment from
    the previous step.

    Args:
        image_path: Path to the target image.
        model: Loaded HomographyNet.
        reference: (H, W) float32 grayscale reference at working resolution.
        cfg: Config dict.
        device: Torch device.
        num_iterations: Number of predict-warp-predict iterations (default 1).

    Returns:
        (H_full, four_point) where H_full is the (3, 3) composed homography at
        full resolution and four_point is the last iteration's displacement.
    """
    hcfg = cfg["homography"]
    working_w, working_h = hcfg["working_width"], hcfg["working_height"]
    working_size = (working_w, working_h)

    # Load and preprocess target
    target_bgr = cv2.imread(str(image_path))
    if target_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig_h, orig_w = target_bgr.shape[:2]

    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    target_working = cv2.resize(target_gray, (working_w, working_h)).astype(np.float32) / 255.0

    # Iterative prediction
    H_composed = np.eye(3, dtype=np.float64)
    current_target = target_working

    for i in range(num_iterations):
        input_pair = np.stack([reference, current_target], axis=0)
        input_tensor = torch.from_numpy(input_pair).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_four_point = model(input_tensor).cpu().numpy().flatten()

        H_iter = four_point_to_homography(pred_four_point, working_size)
        H_composed = H_iter @ H_composed

        if i < num_iterations - 1:
            # Warp original target by composed H for next iteration
            current_target = warp_image(target_working, H_composed, working_size)

    # Scale to full resolution
    H_full = scale_homography(H_composed, working_size, (orig_w, orig_h))

    return H_full, pred_four_point


def warp_to_reference(
    image_path: str,
    H: np.ndarray,
    output_size: tuple[int, int] = None,
) -> np.ndarray:
    """Warp a full-resolution image to the reference frame.

    Args:
        image_path: Path to the target image.
        H: (3, 3) homography at full resolution.
        output_size: (width, height) of output. Defaults to image size.

    Returns:
        Warped BGR image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_size is None:
        h, w = image.shape[:2]
        output_size = (w, h)

    return warp_image(image, H, output_size)


def predict_and_warp(
    image_path: str,
    config_path: str = "configs/config.yaml",
    checkpoint_path: str = None,
    save_path: str = None,
) -> dict:
    """Full pipeline: predict homography and warp image.

    Args:
        image_path: Path to target image.
        config_path: Path to config yaml.
        checkpoint_path: Path to model checkpoint.
        save_path: If provided, save warped image here.

    Returns:
        Dict with H_full, four_point, and optionally warped_path.
    """
    model, reference, mask, cfg, device = load_homography_model(config_path, checkpoint_path)

    H_full, four_point = predict_homography(image_path, model, reference, cfg, device)
    warped = warp_to_reference(image_path, H_full)

    result = {
        "filename": Path(image_path).name,
        "H": H_full.tolist(),
        "four_point": four_point.tolist(),
    }

    if save_path:
        cv2.imwrite(save_path, warped)
        result["warped_path"] = save_path
        logger.info(f"Saved warped image to {save_path}")

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths to align")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="results/homography")
    args = parser.parse_args()

    model, reference, mask, cfg, device = load_homography_model(args.config, args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in args.images:
        try:
            H_full, four_point = predict_homography(img_path, model, reference, cfg, device)
            warped = warp_to_reference(img_path, H_full)

            save_name = f"aligned_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(output_dir / save_name), warped)

            results.append({
                "filename": Path(img_path).name,
                "four_point": four_point.tolist(),
                "warped": save_name,
            })
            logger.info(f"Aligned: {img_path} -> {save_name}")
        except Exception as e:
            logger.error(f"Failed on {img_path}: {e}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
