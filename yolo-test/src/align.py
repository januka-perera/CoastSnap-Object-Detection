"""
Image alignment and plan-view rectification via homography.

Process
-------
All virtual GCPs lie on the z=0 plane (ENU coordinates, metres).
Because all points are coplanar, a 2D homography is the correct and
sufficient tool — full 3D pose estimation is not used.

DELIVERABLE 1 — Register query image to reference image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. H_w2r = findHomography(world_xy, uv_ref)   world → reference image
2. H_w2q = findHomography(world_xy, uv_q)     world → query image
3. H_q2r = H_w2r @ inv(H_w2q)                query → reference image
4. aligned = warpPerspective(query, H_q2r, ref_size)

DELIVERABLE 2 — North-up metric plan view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Define output grid: xlim, ylim (local metres), dx (metres/pixel)
2. World-to-plan transform M:
       u_plan = (X - xmin) / dx
       v_plan = (ymax - Y) / dx
3. H_w2q = findHomography(world_xy, uv_q)
4. H_i2p = M @ inv(H_w2q)
5. plan  = warpPerspective(query, H_i2p, (nx, ny))

reference_keypoints.json format
--------------------------------
    {
        "sign":       {"image": [u, v], "world": [X, Y, Z]},
        "building-1": {"image": [u, v], "world": [X, Y, Z]},
        ...
    }
    image : 2D pixel coords in the reference image
    world : local ENU metres; only X and Y are used (z=0 plane)

Usage
-----
    python align.py \\
        --reference      ../../data/images/reference.jpg \\
        --reference-keypoints ../reference/reference.json \\
        --yolo-weights   ../yolo_runs/phase3_full/weights/best.pt \\
        --kp-weights-dir ../keypoint_checkpoints \\
        --images-dir     ../../data/images \\
        --output-dir     ../aligned \\
        --rectify \\
        --xlim -300 500 --ylim -1600 -400
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
# Reference keypoints
# ---------------------------------------------------------------------------

def load_reference_keypoints(path: str):
    """
    Load reference keypoints from JSON.

    Returns
    -------
    class_names   : ordered list of landmark class names
    world_xy      : (N, 2) float32 — XY world coordinates (z ignored)
    ref_image_pts : (N, 2) float32 — pixel coords in the reference image
    """
    with open(path) as f:
        data = json.load(f)
    class_names   = []
    world_xy      = []
    ref_image_pts = []
    for cls, val in data.items():
        class_names.append(cls)
        world_xy.append(val["world"][:2])      # take only X, Y
        ref_image_pts.append(val["image"])
    return (
        class_names,
        np.array(world_xy,      dtype=np.float32),
        np.array(ref_image_pts, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# World-to-plan affine transform
# ---------------------------------------------------------------------------

def world_to_plan_matrix(xlim: tuple, ylim: tuple, dx: float) -> tuple:
    """
    Build the 3×3 projective matrix M that maps world (X, Y) coordinates
    to plan-view pixel coordinates.

        u_plan = (X - xmin) / dx
        v_plan = (ymax - Y) / dx   (north-up: row 0 = ymax)

    Returns
    -------
    M        : (3, 3) float64
    (nx, ny) : output image size in pixels
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    nx = int(round((xmax - xmin) / dx)) + 1
    ny = int(round((ymax - ymin) / dx)) + 1
    M = np.array(
        [[1 / dx,      0,    -xmin / dx],
         [0,      -1 / dx,   ymax / dx],
         [0,           0,            1]],
        dtype=np.float64,
    )
    return M, (nx, ny)


# ---------------------------------------------------------------------------
# Per-image alignment (and optional plan-view rectification)
# ---------------------------------------------------------------------------

def align_image(
    query_path: str,
    class_names: list,
    world_xy: np.ndarray,
    detected_2d: dict,
    H_w2r: np.ndarray,
    ref_size: tuple,
    output_path: str,
    min_points: int = 4,
    ransac_threshold: float = 5.0,
    rectify_path: str = None,
    M: np.ndarray = None,
    plan_size: tuple = None,
) -> bool:
    """
    Register a query image to the reference and optionally produce a
    north-up plan-view image.

    Parameters
    ----------
    class_names      : ordered landmark names matching world_xy rows.
    world_xy         : (N, 2) world XY coords of landmarks (local metres).
    detected_2d      : dict mapping class_name → [kp_x, kp_y] in query.
    H_w2r            : (3, 3) world → reference image homography.
    ref_size         : (width, height) of the reference image.
    min_points       : minimum inlier count required to proceed.
    ransac_threshold : reprojection threshold in pixels for RANSAC.
    rectify_path     : if set, also save a plan-view image here.
    M                : (3, 3) world-to-plan transform (from world_to_plan_matrix).
    plan_size        : (nx, ny) plan-view output size in pixels.
    """
    stem = Path(query_path).name

    # ── Match detected keypoints to world coords in consistent order ──────
    matched_world = []
    matched_uv_q  = []

    for i, cls in enumerate(class_names):
        if cls in detected_2d:
            matched_world.append(world_xy[i])
            matched_uv_q.append(detected_2d[cls])

    if len(matched_world) < min_points:
        missing = [c for c in class_names if c not in detected_2d]
        print(
            f"  [SKIP] {stem}: {len(matched_world)}/{len(class_names)} landmarks "
            f"detected — missing {missing}"
        )
        return False

    src_world = np.array(matched_world, dtype=np.float32)
    src_uv_q  = np.array(matched_uv_q,  dtype=np.float32)

    # ── H_w2q: world → query image ────────────────────────────────────────
    H_w2q, mask = cv2.findHomography(src_world, src_uv_q, cv2.RANSAC, ransac_threshold)

    if H_w2q is None:
        print(f"  [SKIP] {stem}: findHomography failed")
        return False

    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < min_points:
        print(f"  [SKIP] {stem}: only {n_inliers} inliers after RANSAC")
        return False

    print(f"  {stem}: inliers={n_inliers}/{len(matched_world)}")

    # ── Load query image ──────────────────────────────────────────────────
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"  [SKIP] {stem}: cannot read image")
        return False

    # ── Deliverable 1: align query to reference ───────────────────────────
    # H_q2r = H_w2r @ inv(H_w2q)
    H_q2r = H_w2r @ np.linalg.inv(H_w2q)
    ref_W, ref_H = ref_size
    aligned = cv2.warpPerspective(query_img, H_q2r, (ref_W, ref_H))
    cv2.imwrite(output_path, aligned)

    # ── Deliverable 2: plan-view rectification ────────────────────────────
    # H_i2p = M @ inv(H_w2q)  maps query pixels → plan pixels
    if rectify_path is not None:
        if M is None or plan_size is None:
            print(f"  [WARN] {stem}: plan-view params missing, skipping")
        else:
            H_i2p = M @ np.linalg.inv(H_w2q)
            nx, ny = plan_size
            plan = cv2.warpPerspective(query_img, H_i2p, (nx, ny))
            cv2.imwrite(rectify_path, plan)
            print(f"  {stem}: plan-view saved → {rectify_path}")

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Align query images to a reference via z=0 plane homography"
    )
    _default_cfg = str(Path(__file__).parent.parent / "configs" / "config.yaml")

    parser.add_argument("--config",              default=_default_cfg)
    parser.add_argument("--reference",           required=True,
                        help="Path to the reference image.")
    parser.add_argument("--reference-keypoints", required=True,
                        help="JSON file with 'image' and 'world' coords per landmark.")
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
                        help="Minimum RANSAC inliers required (default 4).")
    parser.add_argument("--ransac-threshold",    type=float, default=5.0,
                        help="RANSAC reprojection threshold in pixels (default 5).")
    parser.add_argument("--no-subpixel",         action="store_true")

    # ── Plan-view rectification ───────────────────────────────────────────
    parser.add_argument("--rectify",             action="store_true",
                        help="Also produce a north-up metric plan-view image.")
    parser.add_argument("--rectify-dir",         default="./rectified",
                        help="Output directory for plan-view images (default: ./rectified).")
    parser.add_argument("--xlim",                type=float, nargs=2,
                        metavar=("X_MIN", "X_MAX"),
                        help="East bounds in local metres. E.g. --xlim -300 500")
    parser.add_argument("--ylim",                type=float, nargs=2,
                        metavar=("Y_MIN", "Y_MAX"),
                        help="North bounds in local metres. E.g. --ylim -1600 -400")
    parser.add_argument("--dx",                  type=float, default=0.5,
                        help="Plan-view resolution in metres per pixel (default 0.5).")

    args = parser.parse_args()

    if args.kp_weights is None and args.kp_weights_dir is None:
        parser.error("Provide --kp-weights or --kp-weights-dir.")
    if args.image is None and args.images_dir is None:
        parser.error("Provide --image or --images-dir.")
    if args.rectify and (args.xlim is None or args.ylim is None):
        parser.error("--rectify requires --xlim and --ylim.")

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Reference image ───────────────────────────────────────────────────
    ref_img = cv2.imread(args.reference)
    if ref_img is None:
        print(f"ERROR: cannot read reference image: {args.reference}")
        sys.exit(1)
    ref_H_px, ref_W_px = ref_img.shape[:2]
    ref_size = (ref_W_px, ref_H_px)
    print(f"Reference image : {args.reference}  ({ref_W_px}×{ref_H_px})")

    # ── Reference keypoints ───────────────────────────────────────────────
    class_names, world_xy, ref_image_pts = load_reference_keypoints(
        args.reference_keypoints
    )
    print(f"Landmarks       : {class_names}")

    # Subtract the centroid of the world coordinates for numerical stability.
    # UTM eastings/northings (~341 000, ~6 259 000) cause precision loss in
    # the DLT solver.  xlim/ylim must be given in the same local frame
    # (i.e. UTM minus this origin).
    xy_origin = world_xy.mean(axis=0)
    world_xy  = world_xy - xy_origin
    print(
        f"UTM XY origin   : ({xy_origin[0]:.2f}, {xy_origin[1]:.2f})  "
        f"(subtracted for numerical stability)"
    )

    # ── H_w2r: world → reference image ───────────────────────────────────
    print("\nEstimating world→reference homography…")
    H_w2r, mask_ref = cv2.findHomography(
        world_xy, ref_image_pts, cv2.RANSAC, args.ransac_threshold
    )
    if H_w2r is None:
        print("ERROR: could not estimate reference homography.")
        sys.exit(1)
    n_ref_inliers = int(mask_ref.sum()) if mask_ref is not None else 0
    print(f"  inliers={n_ref_inliers}/{len(world_xy)}")

    # ── Plan-view setup ───────────────────────────────────────────────────
    M         = None
    plan_size = None
    rectify_dir = None

    if args.rectify:
        rectify_dir = Path(args.rectify_dir)
        rectify_dir.mkdir(parents=True, exist_ok=True)
        xlim = tuple(args.xlim)
        ylim = tuple(args.ylim)
        M, plan_size = world_to_plan_matrix(xlim, ylim, args.dx)
        nx, ny = plan_size
        print(
            f"\n[rectify] xlim={xlim}  ylim={ylim}  dx={args.dx} m/px"
            f"  →  {nx}×{ny} px  output: {rectify_dir}\n"
        )

    # ── Load detection models ─────────────────────────────────────────────
    yolo_model, _, _ = load_yolo(
        args.yolo_weights,
        cfg["yolo"]["conf_threshold"],
        cfg["yolo"]["iou_threshold"],
    )
    decoder_channels = cfg["keypoint"].get("decoder_channels", [128, 64, 32])
    if args.kp_weights_dir:
        kp_models = load_per_class_models(
            args.kp_weights_dir,
            cfg["data"].get("classes", []),
            decoder_channels,
            device,
        )
    else:
        kp_models = load_keypoint_model(args.kp_weights, decoder_channels, device)

    # ── Collect query image paths ─────────────────────────────────────────
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
            if p.resolve() != ref_resolved
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each query image ──────────────────────────────────────────
    print(f"\nProcessing {len(image_paths)} query image(s) → {out_dir}\n")

    n_success = 0
    for img_path in image_paths:
        stem = Path(img_path).stem

        results     = run_pipeline(
            img_path, yolo_model, kp_models, cfg, device,
            subpixel=not args.no_subpixel,
        )
        detected_2d = {r.detection.class_name: [r.kp_x, r.kp_y] for r in results}

        out_path     = str(out_dir / f"{stem}_aligned.jpg")
        rectify_path = str(rectify_dir / f"{stem}_plan.jpg") if rectify_dir else None

        success = align_image(
            query_path=img_path,
            class_names=class_names,
            world_xy=world_xy,
            detected_2d=detected_2d,
            H_w2r=H_w2r,
            ref_size=ref_size,
            output_path=out_path,
            min_points=args.min_points,
            ransac_threshold=args.ransac_threshold,
            rectify_path=rectify_path,
            M=M,
            plan_size=plan_size,
        )
        if success:
            n_success += 1

    print(f"\nDone.  {n_success}/{len(image_paths)} images aligned successfully.")


if __name__ == "__main__":
    main()
