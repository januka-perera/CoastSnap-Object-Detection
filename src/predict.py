"""Inference on new images."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .model import HeatmapRegressor
from .heatmap_utils import soft_argmax_coordinates, argmax_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ImageNet stats
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(
    config_path: str = "configs/config.yaml",
    checkpoint_path: str = None,
) -> tuple:
    """Load model and config.

    Returns:
        (model, cfg, device)
    """
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

    return model, cfg, device


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Run inference on a single image.

    Args:
        image_path: Path to the input image.
        model: Loaded model.
        cfg: Config dict.
        device: Torch device.

    Returns:
        Dict with 'landmarks' list of {id, u, v, confidence}.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    input_w = cfg["data"]["input_width"]
    input_h = cfg["data"]["input_height"]

    # Preprocess
    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    normalised = resized.astype(np.float32) / 255.0
    normalised = (normalised - MEAN) / STD
    tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        heatmaps = model(tensor)

    use_soft = cfg["inference"]["soft_argmax"]
    window_size = cfg["inference"]["local_window_size"]

    if use_soft:
        coords = soft_argmax_coordinates(heatmaps, window_size)
    else:
        coords = argmax_coordinates(heatmaps)

    coords = coords.cpu().squeeze(0)  # (N, 2)
    heatmaps_np = heatmaps.cpu().squeeze(0).numpy()  # (N, H, W)

    landmarks = []
    for n in range(cfg["data"]["num_landmarks"]):
        u = coords[n, 0].item() * orig_w
        v = coords[n, 1].item() * orig_h
        confidence = float(heatmaps_np[n].max())
        landmarks.append({
            "id": n,
            "u": round(u, 1),
            "v": round(v, 1),
            "confidence": round(confidence, 4),
        })

    return {
        "filename": Path(image_path).name,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "landmarks": landmarks,
    }


def predict_batch(
    image_paths: list[str],
    config_path: str = "configs/config.yaml",
    checkpoint_path: str = None,
) -> list[dict]:
    """Run inference on multiple images.

    Args:
        image_paths: List of image file paths.
        config_path: Path to config yaml.
        checkpoint_path: Path to model checkpoint.

    Returns:
        List of prediction dicts.
    """
    model, cfg, device = load_model(config_path, checkpoint_path)

    results = []
    for path in image_paths:
        try:
            result = predict_image(path, model, cfg, device)
            results.append(result)
            logger.info(f"Predicted: {path}")
        except Exception as e:
            logger.error(f"Failed on {path}: {e}")
            results.append({"filename": Path(path).name, "error": str(e)})

    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths to predict on")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = predict_batch(args.images, args.config, args.checkpoint)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved predictions to {args.output}")
    else:
        print(json.dumps(results, indent=2))
