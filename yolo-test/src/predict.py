"""
Full inference pipeline: YOLO detection → keypoint prediction.

Steps
-----
1. Load image.
2. Run YOLO detection; filter by confidence threshold.
3. Select top-k detections (default: top-1 by confidence).
4. For each detection:
   a. Crop detected region from original image.
   b. Resize crop to model input size (224 × 224).
   c. Predict keypoint heatmap.
   d. Extract heatmap peak coordinate.
   e. Map coordinate back to original image space.
5. Save / display annotated image.

Usage
-----
    # Single image
    python predict.py --image path/to/img.jpg \
                      --yolo-weights ./yolo_runs/phase3_full/weights/best.pt \
                      --kp-weights   ./keypoint_checkpoints/keypoint_best.pt

    # Directory of images
    python predict.py --images-dir ./data/images \
                      --yolo-weights ... --kp-weights ...
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model         import KeypointHeatmapModel
from heatmap_utils import extract_coordinate

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class KeypointResult:
    detection:  Detection
    kp_x:       float        # absolute pixel x in original image
    kp_y:       float        # absolute pixel y in original image
    kp_x_norm:  float        # normalised x in [0, 1]
    kp_y_norm:  float        # normalised y in [0, 1]
    heatmap:    np.ndarray   # (hm_h, hm_w) predicted heatmap


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_yolo(weights: str, conf: float, iou: float):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        sys.exit(1)
    return YOLO(weights), conf, iou


def load_keypoint_model(
    weights: str,
    decoder_channels: list,
    device: torch.device,
) -> KeypointHeatmapModel:
    model = KeypointHeatmapModel(
        pretrained=False,
        decoder_channels=decoder_channels,
        freeze_backbone=False,
    )
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval().to(device)
    return model


def load_per_class_models(
    kp_weights_dir: str,
    class_names: list,
    decoder_channels: list,
    device: torch.device,
) -> dict:
    """
    Load one keypoint model per class from *kp_weights_dir*.

    For each class name, looks for  ``{kp_weights_dir}/{class_name}_best.pt``.
    Falls back to ``keypoint_best.pt`` if a per-class file is missing.

    Returns
    -------
    dict mapping class_name → KeypointHeatmapModel
    """
    weights_dir = Path(kp_weights_dir)
    fallback    = weights_dir / "keypoint_best.pt"
    models: dict = {}
    for name in class_names:
        candidate = weights_dir / f"{name}_best.pt"
        if candidate.exists():
            path = candidate
        elif fallback.exists():
            print(f"  [WARN] No per-class weights for '{name}', using {fallback.name}")
            path = fallback
        else:
            print(f"  [WARN] No weights found for '{name}' — skipping")
            continue
        print(f"  Loading keypoint model for '{name}': {path}")
        models[name] = load_keypoint_model(str(path), decoder_channels, device)
    return models


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(
    yolo_model,
    image: np.ndarray,
    conf: float,
    iou: float,
    class_names: list,
) -> list:
    results     = yolo_model(image, conf=conf, iou=iou, verbose=False)
    detections  = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            name   = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append(
                Detection(cls_id, name, float(box.conf.item()), x1, y1, x2, y2)
            )
    return detections


# ---------------------------------------------------------------------------
# Keypoint prediction on one crop
# ---------------------------------------------------------------------------

def predict_keypoint(
    kp_model: KeypointHeatmapModel,
    image_bgr: np.ndarray,
    det: Detection,
    input_size: tuple,
    heatmap_size: tuple,
    device: torch.device,
    subpixel: bool = True,
):
    H, W   = image_bgr.shape[:2]
    in_h, in_w = input_size
    hm_h, hm_w = heatmap_size

    # Clamp crop bounds
    x1 = max(0, det.x1);  y1 = max(0, det.y1)
    x2 = min(W, det.x2);  y2 = min(H, det.y2)
    crop_h = y2 - y1;     crop_w = x2 - x1

    if crop_h == 0 or crop_w == 0:
        return None

    # Preprocess crop
    crop  = image_bgr[y1:y2, x1:x2]
    crop  = cv2.resize(crop, (in_w, in_h))
    crop  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop  = (crop - _IMAGENET_MEAN) / _IMAGENET_STD
    t     = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        hm = kp_model(t)                              # (1, 1, hm_h, hm_w)
    hm_np = hm.squeeze().cpu().numpy()               # (hm_h, hm_w)

    # Extract normalised coordinate in heatmap space
    kp_xn_hm, kp_yn_hm = extract_coordinate(hm_np, hm_h, hm_w, subpixel=subpixel)

    # Map to original crop pixel space (accounting for resize)
    kp_x_crop = kp_xn_hm * crop_w
    kp_y_crop = kp_yn_hm * crop_h

    # Map to original image space
    kp_x = x1 + kp_x_crop
    kp_y = y1 + kp_y_crop

    return KeypointResult(
        detection=det,
        kp_x=kp_x,     kp_y=kp_y,
        kp_x_norm=kp_x / W,
        kp_y_norm=kp_y / H,
        heatmap=hm_np,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    yolo_model,
    kp_models,
    cfg: dict,
    device: torch.device,
    subpixel: bool = True,
) -> list:
    """
    Run full detection + keypoint pipeline on one image.

    For each class, selects the single highest-confidence detection and
    predicts its keypoint.  With two classes (sign, building-1) this
    produces exactly two KeypointResult objects per image.

    Parameters
    ----------
    kp_models : Either a single KeypointHeatmapModel (applied to every detection)
                or a dict mapping class_name → KeypointHeatmapModel for per-class routing.
    """
    yolo_cfg    = cfg["yolo"]
    kp_cfg      = cfg["keypoint"]
    data_cfg    = cfg["data"]
    class_names = data_cfg.get("classes", [])

    # Normalise kp_models to a dict for uniform handling
    if not isinstance(kp_models, dict):
        kp_models = {"*": kp_models}   # wildcard: same model for every class

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: cannot read {image_path}")
        return []

    dets = detect(
        yolo_model, image,
        conf=yolo_cfg["conf_threshold"],
        iou=yolo_cfg["iou_threshold"],
        class_names=class_names,
    )

    if not dets:
        if yolo_cfg.get("fallback_full_image", False):
            H, W = image.shape[:2]
            fallback_det = Detection(
                class_id=0, class_name=class_names[0] if class_names else "sign",
                confidence=0.0,
                x1=0, y1=0, x2=W, y2=H,
            )
            print(f"  [WARN] No detection in {image_path} — using full image fallback")
            dets = [fallback_det]
        else:
            return []

    # Always keep the single highest-confidence detection per class.
    # This means if both "sign" and "building-1" are detected, both are
    # processed independently — one keypoint prediction each.
    best_per_class: dict = {}
    for d in dets:
        if (d.class_name not in best_per_class
                or d.confidence > best_per_class[d.class_name].confidence):
            best_per_class[d.class_name] = d
    dets = list(best_per_class.values())

    input_size   = tuple(kp_cfg["input_size"])
    heatmap_size = tuple(kp_cfg["heatmap_size"])

    results = []
    for det in dets:
        # Route to per-class model, then wildcard, then first available
        model = (
            kp_models.get(det.class_name)
            or kp_models.get("*")
            or next(iter(kp_models.values()), None)
        )
        if model is None:
            print(f"  [WARN] No keypoint model for class '{det.class_name}' — skipping")
            continue
        r = predict_keypoint(
            model, image, det,
            input_size=input_size,
            heatmap_size=heatmap_size,
            device=device,
            subpixel=subpixel,
        )
        if r is not None:
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(
    image_path: str,
    results: list,
    output_path: str = None,
    show: bool = False,
) -> np.ndarray:
    img = cv2.imread(image_path)

    for res in results:
        d = res.detection
        # Bounding box
        cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), (0, 220, 0), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(img, label, (d.x1, max(d.y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)
        # Keypoint
        kx, ky = int(round(res.kp_x)), int(round(res.kp_y))
        cv2.circle(img, (kx, ky), 6,  (0, 0, 255), -1)
        cv2.circle(img, (kx, ky), 8,  (255, 255, 255), 1)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")

    if show:
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLO + keypoint inference pipeline")
    _default_cfg = str(Path(__file__).parent.parent / "configs" / "config.yaml")
    parser.add_argument("--config",         default=_default_cfg)
    parser.add_argument("--image",          default=None,  help="Single image path.")
    parser.add_argument("--images-dir",     default=None,  help="Directory of images.")
    parser.add_argument("--yolo-weights",   required=True, help="YOLO .pt weights.")
    parser.add_argument("--kp-weights",     default=None,
                        help="Single keypoint model .pt (used for all classes).")
    parser.add_argument("--kp-weights-dir", default=None,
                        help="Directory containing per-class checkpoints "
                             "named <class_name>_best.pt.")
    parser.add_argument("--output-dir",     default="./predictions")
    parser.add_argument("--no-subpixel",  action="store_true")
    parser.add_argument("--show",         action="store_true",
                        help="Display each prediction interactively.")
    args = parser.parse_args()

    if args.kp_weights is None and args.kp_weights_dir is None:
        parser.error("Provide --kp-weights (single model) or --kp-weights-dir (per-class).")

    cfg    = load_config(args.config)
    kp_cfg = cfg["keypoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    yolo_model, _, _ = load_yolo(
        args.yolo_weights,
        cfg["yolo"]["conf_threshold"],
        cfg["yolo"]["iou_threshold"],
    )

    decoder_channels = kp_cfg.get("decoder_channels", [128, 64, 32])

    if args.kp_weights_dir:
        class_names = cfg["data"].get("classes", [])
        kp_models = load_per_class_models(
            args.kp_weights_dir, class_names, decoder_channels, device
        )
        if not kp_models:
            print("ERROR: No keypoint models loaded from --kp-weights-dir.")
            sys.exit(1)
    else:
        kp_models = load_keypoint_model(
            args.kp_weights, decoder_channels=decoder_channels, device=device
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    elif args.images_dir:
        exts        = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_paths = []
        for ext in exts:
            image_paths.extend(Path(args.images_dir).glob(ext))
        image_paths = sorted(str(p) for p in image_paths)
    else:
        print("ERROR: provide --image or --images-dir")
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s)…")

    for img_path in image_paths:
        results  = run_pipeline(
            img_path, yolo_model, kp_models, cfg, device,
            subpixel=not args.no_subpixel,
        )
        stem     = Path(img_path).stem
        out_path = str(out_dir / f"{stem}_pred.jpg")
        visualise(img_path, results, output_path=out_path, show=args.show)

        for res in results:
            d = res.detection
            print(
                f"  {stem}  |  {d.class_name}  conf={d.confidence:.2f}"
                f"  kp=({res.kp_x:.1f}, {res.kp_y:.1f})"
            )


if __name__ == "__main__":
    main()
