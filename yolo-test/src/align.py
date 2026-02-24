"""
Image alignment to a reference using predicted keypoint homography.

Steps
-----
1. Load reference keypoints (manually specified pixel coordinates in the
   reference image — no inference is run on the reference).
2. Run YOLO + keypoint pipeline on each query image to predict keypoints
   in query pixel space.
3. Match predicted keypoints to reference keypoints by class name.
4. If ≥ min_points matches found, compute homography with RANSAC.
5. Warp query image into reference image coordinate space.

Resolution handling
-------------------
Reference keypoints are in reference image pixel space (W_ref × H_ref).
Predicted keypoints are in query image pixel space (W_q × H_q).
cv2.findHomography maps query → reference in absolute pixel coordinates,
so different resolutions are handled naturally.  The output image is always
(W_ref × H_ref) regardless of the query image resolution.

Usage
-----
    # Directory of images
    python align.py \\
        --reference      ../../data/images/reference.jpg \\
        --reference-keypoints ../reference_keypoints.json \\
        --yolo-weights   ../yolo_runs/phase3_full/weights/best.pt \\
        --kp-weights-dir ../keypoint_checkpoints \\
        --images-dir     ../../data/images \\
        --output-dir     ../aligned

    # Single image
    python align.py \\
        --reference      ../../data/images/reference.jpg \\
        --reference-keypoints ../reference_keypoints.json \\
        --yolo-weights   ../yolo_runs/phase3_full/weights/best.pt \\
        --kp-weights-dir ../keypoint_checkpoints \\
        --image          ../../data/images/query.jpg

reference_keypoints.json format (pixel coords in the reference image):
    {
        "sign":       [312, 187],
        "building-1": [445, 223],
        "pole":       [100, 350],
        "rock":       [600, 420]
    }
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from predict import (
    load_config,
    load_yolo,
    load_keypoint_model,
    load_per_class_models,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_reference_keypoints(path: str) -> dict:
    """
    Load manually specified reference keypoints from a JSON file.

    Returns
    -------
    dict  :  class_name → (x, y) in reference image pixel coordinates
    """
    with open(path) as f:
        data = json.load(f)
    # Accept both [x, y] lists and {"x": ..., "y": ...} dicts
    result = {}
    for cls, val in data.items():
        if isinstance(val, (list, tuple)):
            result[cls] = (float(val[0]), float(val[1]))
        elif isinstance(val, dict):
            result[cls] = (float(val["x"]), float(val["y"]))
        else:
            raise ValueError(f"Unrecognised keypoint format for '{cls}': {val}")
    return result


def align_image(
    query_path: str,
    ref_keypoints: dict,
    ref_size: tuple,
    yolo_model,
    kp_models,
    cfg: dict,
    device: torch.device,
    output_path: str,
    min_points: int = 4,
    ransac_reproj_threshold: float = 5.0,
    subpixel: bool = True,
) -> bool:
    """
    Predict keypoints in the query image, compute homography against the
    reference keypoints, and warp the query image to reference space.

    Parameters
    ----------
    ref_size                  : (width, height) of the reference image.
                                The aligned output will have this size.
    min_points                : Minimum matched keypoints required to attempt
                                homography.  Must be ≥ 4.
    ransac_reproj_threshold   : Maximum reprojection error (pixels) for a
                                point pair to be considered an inlier by RANSAC.

    Returns
    -------
    True if the image was aligned and saved, False otherwise.
    """
    stem = Path(query_path).name

    # ── Run detection + keypoint pipeline ────────────────────────────────
    results = run_pipeline(
        query_path, yolo_model, kp_models, cfg, device, subpixel=subpixel
    )

    if not results:
        print(f"  [SKIP] {stem}: no detections")
        return False

    # ── Match predicted keypoints to reference by class name ─────────────
    # src_pts: query image pixel coords  (what we detected)
    # dst_pts: reference image pixel coords (manually specified)
    src_pts        = []
    dst_pts        = []
    matched_classes = []

    for res in results:
        cls = res.detection.class_name
        if cls in ref_keypoints:
            # kp_x / kp_y are absolute pixels in the query image
            src_pts.append([res.kp_x, res.kp_y])
            dst_pts.append(list(ref_keypoints[cls]))
            matched_classes.append(cls)

    if len(src_pts) < min_points:
        print(
            f"  [SKIP] {stem}: {len(src_pts)}/{min_points} keypoints matched "
            f"(need {min_points}) — found {matched_classes}"
        )
        return False

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # ── Compute homography (RANSAC) ───────────────────────────────────────
    # src (query pixel space) → dst (reference pixel space)
    # Both sets of coordinates are absolute pixels in their respective images,
    # so different resolutions are handled correctly.
    M, mask = cv2.estimateAffine2D(
        src_pts, dst_pts,
        cv2.RANSAC
    )

    if M is None:
        print(f"  [SKIP] {stem}: findHomography returned None (degenerate configuration?)")
        return False

    n_inliers = int(mask.sum()) if mask is not None else len(src_pts)
    print(
        f"  {stem}: matched={matched_classes}  "
        f"inliers={n_inliers}/{len(src_pts)}"
    )

    if n_inliers < min_points:
        print(f"  [SKIP] {stem}: too few RANSAC inliers ({n_inliers}/{min_points})")
        return False

    # ── Warp query image → reference coordinate space ────────────────────
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"  [SKIP] {stem}: cannot read image")
        return False

    ref_W, ref_H = ref_size
    aligned = cv2.warpAffine(query_img, M, (ref_W, ref_H))

    cv2.imwrite(output_path, aligned)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Align query images to a reference image via keypoint homography"
    )
    _default_cfg = str(Path(__file__).parent.parent / "configs" / "config.yaml")

    parser.add_argument("--config",              default=_default_cfg)
    parser.add_argument("--reference",           required=True,
                        help="Path to the reference image.")
    parser.add_argument("--reference-keypoints", required=True,
                        help="JSON file with manually specified keypoints "
                             "in the reference image.")
    parser.add_argument("--image",               default=None,
                        help="Single query image to align.")
    parser.add_argument("--images-dir",          default=None,
                        help="Directory of query images to align.")
    parser.add_argument("--yolo-weights",        required=True)
    parser.add_argument("--kp-weights",          default=None,
                        help="Single keypoint model .pt (used for all classes).")
    parser.add_argument("--kp-weights-dir",      default=None,
                        help="Directory of per-class keypoint models "
                             "(<class_name>_best.pt).")
    parser.add_argument("--output-dir",          default="./aligned")
    parser.add_argument("--min-points",          type=int, default=4,
                        help="Minimum keypoint matches required for homography (≥4).")
    parser.add_argument("--ransac-threshold",    type=float, default=5.0,
                        help="RANSAC reprojection error threshold in pixels.")
    parser.add_argument("--no-subpixel",         action="store_true")
    args = parser.parse_args()

    if args.kp_weights is None and args.kp_weights_dir is None:
        parser.error("Provide --kp-weights (single model) or --kp-weights-dir (per-class).")
    if args.image is None and args.images_dir is None:
        parser.error("Provide --image or --images-dir.")
    if args.min_points < 4:
        parser.error("--min-points must be ≥ 4 for homography.")

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Reference image size ──────────────────────────────────────────────
    ref_img = cv2.imread(args.reference)
    if ref_img is None:
        print(f"ERROR: Cannot read reference image: {args.reference}")
        sys.exit(1)
    ref_H_px, ref_W_px = ref_img.shape[:2]
    print(f"Reference image  : {args.reference}  ({ref_W_px}×{ref_H_px})")

    # ── Reference keypoints ───────────────────────────────────────────────
    ref_keypoints = load_reference_keypoints(args.reference_keypoints)
    print(f"Reference classes: {list(ref_keypoints.keys())}")

    # ── Load models ───────────────────────────────────────────────────────
    yolo_model, _, _ = load_yolo(
        args.yolo_weights,
        cfg["yolo"]["conf_threshold"],
        cfg["yolo"]["iou_threshold"],
    )

    decoder_channels = cfg["keypoint"].get("decoder_channels", [128, 64, 32])
    if args.kp_weights_dir:
        class_names = cfg["data"].get("classes", [])
        kp_models   = load_per_class_models(
            args.kp_weights_dir, class_names, decoder_channels, device
        )
    else:
        kp_models = load_keypoint_model(args.kp_weights, decoder_channels, device)

    # ── Collect query images ──────────────────────────────────────────────
    ref_resolved = Path(args.reference).resolve()
    if args.image:
        image_paths = [args.image]
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_paths = []
        for ext in exts:
            image_paths.extend(Path(args.images_dir).glob(ext))
        image_paths = sorted(
            str(p) for p in image_paths
            if p.resolve() != ref_resolved   # exclude the reference itself
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAligning {len(image_paths)} image(s) → {out_dir}\n")

    n_success = 0
    for img_path in image_paths:
        stem     = Path(img_path).stem
        out_path = str(out_dir / f"{stem}_aligned.jpg")
        success  = align_image(
            query_path=img_path,
            ref_keypoints=ref_keypoints,
            ref_size=(ref_W_px, ref_H_px),
            yolo_model=yolo_model,
            kp_models=kp_models,
            cfg=cfg,
            device=device,
            output_path=out_path,
            min_points=args.min_points,
            ransac_reproj_threshold=args.ransac_threshold,
            subpixel=not args.no_subpixel,
        )
        if success:
            n_success += 1

    print(f"\nDone.  {n_success}/{len(image_paths)} images aligned successfully.")


if __name__ == "__main__":
    main()
