"""
Image alignment to a reference using DLT camera calibration.

Steps
-----
1. Load reference keypoints — manually specified 2D pixel coords in the
   reference image AND 3D world coords (local metres, z=up) for each landmark.
2. Pass 1 — run YOLO + keypoint pipeline on every query image to detect 2D
   keypoint locations.
3. Calibration — jointly estimate the camera intrinsic matrix K (focal length,
   principal point) across all images using cv2.calibrateCamera.
   Assumptions: zero skew, square pixels (fx=fy), no lens distortion.
4. Pass 2 — for each query image, estimate camera pose with solvePnPRansac
   using the calibrated K and known 3D world coordinates.
5. Compute the plane-induced homography for z=0 (water surface) between the
   query and reference cameras, then warp with cv2.warpPerspective.

Why DLT instead of homography?
-------------------------------
Homography requires scene points to be coplanar.  The landmarks here are at
different elevations, so homography is not valid.  DLT estimates the full
projection matrix P = K[R|t] from 2D–3D correspondences and handles
non-coplanar points correctly.  Alignment is performed for the z=0 plane
(the target beach/water surface).

reference_keypoints.json format
--------------------------------
    {
        "sign":       {"image": [1563.36, 1888.30], "world": [X, Y, Z]},
        "building-1": {"image": [2467.44,  652.89], "world": [X, Y, Z]},
        "building-2": {"image": [2768.24,  549.49], "world": [X, Y, Z]},
        "vent":       {"image": [1422.00, 2100.10], "world": [X, Y, Z]},
        "fifth":      {"image": [u, v],             "world": [X, Y, Z]}
    }
    image : 2D pixel coords in the reference image (manually measured)
    world : 3D coords in local metres  (z = elevation, z=0 = water surface)

Usage
-----
    python align.py \\
        --reference      ../../data/images/reference.jpg \\
        --reference-keypoints ../reference/reference.json \\
        --yolo-weights   ../yolo_runs/phase3_full/weights/best.pt \\
        --kp-weights-dir ../keypoint_checkpoints \\
        --images-dir     ../../data/images \\
        --output-dir     ../aligned
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
    world_pts     : (N, 3) float32 — 3D world coordinates in local metres
    ref_image_pts : (N, 2) float32 — 2D pixel coords in the reference image
    """
    with open(path) as f:
        data = json.load(f)
    class_names   = []
    world_pts     = []
    ref_image_pts = []
    for cls, val in data.items():
        class_names.append(cls)
        world_pts.append(val["world"])
        ref_image_pts.append(val["image"])
    return (
        class_names,
        np.array(world_pts,     dtype=np.float32),
        np.array(ref_image_pts, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Camera calibration (DLT)
# ---------------------------------------------------------------------------

def calibrate_camera(
    image_pts_list: list,
    world_pts: np.ndarray,
    image_size: tuple,
) -> tuple:
    """
    Jointly estimate K and per-image poses using cv2.calibrateCamera.

    Assumptions enforced via flags:
      - Zero skew           (OpenCV camera model has no skew term by default)
      - Square pixels fx=fy (CALIB_FIX_ASPECT_RATIO initialised with fx=fy)
      - No lens distortion  (CALIB_FIX_K1/K2/K3 + CALIB_ZERO_TANGENT_DIST)

    Parameters
    ----------
    image_pts_list : list of (N, 2) float32 arrays, one per image.
                     The first element must be the reference image.
    world_pts      : (N, 3) float32 — fixed 3D world coords for every image.
    image_size     : (width, height)

    Returns
    -------
    K     : (3, 3) float64 camera matrix
    rvecs : list of (3, 1) rotation vectors (one per image)
    tvecs : list of (3, 1) translation vectors (one per image)
    """
    obj_pts = [world_pts.reshape(-1, 1, 3)] * len(image_pts_list)
    img_pts = [pts.reshape(-1, 1, 2) for pts in image_pts_list]

    W, H   = image_size
    f_init = float(max(W, H))
    K_init = np.array(
        [[f_init, 0.,      W / 2.],
         [0.,      f_init, H / 2.],
         [0.,      0.,     1.    ]],
        dtype=np.float64,
    )

    flags = (
        cv2.CALIB_FIX_ASPECT_RATIO    # fx = fy  (square pixels)
        | cv2.CALIB_ZERO_TANGENT_DIST # p1 = p2 = 0
        | cv2.CALIB_FIX_K1            # k1 = 0
        | cv2.CALIB_FIX_K2            # k2 = 0
        | cv2.CALIB_FIX_K3            # k3 = 0
    )

    rms, K, _, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size,
        K_init, np.zeros((1, 5), dtype=np.float64),
        flags=flags,
    )
    print(
        f"  Calibration RMS reprojection error : {rms:.3f} px\n"
        f"  f={K[0, 0]:.1f}  cx={K[0, 2]:.1f}  cy={K[1, 2]:.1f}"
    )
    return K, rvecs, tvecs


# ---------------------------------------------------------------------------
# Projection matrix and plane homography
# ---------------------------------------------------------------------------

def projection_matrix(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Build 3×4 projection matrix  P = K [R | t]."""
    R, _ = cv2.Rodrigues(rvec)
    return K @ np.hstack([R, tvec.reshape(3, 1)])


def plane_homography_z0(P_ref: np.ndarray, P_query: np.ndarray) -> np.ndarray:
    """
    Compute the homography induced by the z=0 plane.

    For any point on z=0:  X_world = [X, Y, 0, 1]^T
        p = P * X_world  =  P[:, [0,1,3]] * [X, Y, 1]^T

    The homography mapping query image coords → reference image coords is:
        H = P_ref[:, [0,1,3]] @ inv(P_query[:, [0,1,3]])
    """
    A = P_ref  [:, [0, 1, 3]]
    B = P_query[:, [0, 1, 3]]
    return A @ np.linalg.inv(B)


# ---------------------------------------------------------------------------
# Per-image alignment
# ---------------------------------------------------------------------------

def align_image(
    query_path: str,
    class_names: list,
    world_pts: np.ndarray,
    detected_2d: dict,
    K: np.ndarray,
    P_ref: np.ndarray,
    ref_size: tuple,
    output_path: str,
    min_points: int = 4,
    ransac_reproj_threshold: float = 10.0,
) -> bool:
    """
    Estimate the query camera pose with solvePnPRansac, compute the z=0 plane
    homography to the reference camera, and warp the query image.

    Parameters
    ----------
    detected_2d : dict mapping class_name → [kp_x, kp_y] in query image pixels.
    """
    stem = Path(query_path).name

    # ── Match detected 2D keypoints to 3D world points ────────────────────
    src_2d      = []
    src_3d      = []
    matched_cls = []

    for i, cls in enumerate(class_names):
        if cls in detected_2d:
            src_2d.append(detected_2d[cls])
            src_3d.append(world_pts[i])
            matched_cls.append(cls)

    n_required = len(class_names)
    if len(src_2d) < n_required:
        print(
            f"  [SKIP] {stem}: {len(src_2d)}/{n_required} landmarks detected "
            f"— found {matched_cls}"
        )
        return False

    # ── Load query image and check resolution ────────────────────────────
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"  [SKIP] {stem}: cannot read image")
        return False

    ref_W, ref_H = ref_size
    q_H, q_W    = query_img.shape[:2]

    # K was calibrated in reference image pixel space.  If the query image
    # has a different resolution, scale the detected 2D keypoints to match
    # the reference coordinate system before calling solvePnPRansac.
    scale_x = ref_W / q_W
    scale_y = ref_H / q_H
    if scale_x != 1.0 or scale_y != 1.0:
        src_2d_arr = np.array(src_2d, dtype=np.float32)
        src_2d_arr[:, 0] *= scale_x
        src_2d_arr[:, 1] *= scale_y
        src_2d = src_2d_arr.tolist()

    src_2d = np.array(src_2d, dtype=np.float32).reshape(-1, 1, 2)
    src_3d = np.array(src_3d, dtype=np.float32).reshape(-1, 1, 3)

    # ── Estimate pose with RANSAC ─────────────────────────────────────────
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        src_3d, src_2d, K, None,
        reprojectionError=ransac_reproj_threshold,
        confidence=0.99,
        iterationsCount=1000,
    )

    n_inliers = len(inliers) if inliers is not None else 0
    if not success or n_inliers < min_points:
        print(f"  [SKIP] {stem}: solvePnPRansac failed — {n_inliers}/{len(src_2d)} inliers")
        return False

    print(
        f"  {stem}: matched={matched_cls}  "
        f"inliers={n_inliers}/{len(src_2d)}"
    )

    # ── Plane homography and warp ─────────────────────────────────────────
    # H is computed in reference pixel space.  Resize the query image to
    # reference resolution first so the warp is applied consistently.
    P_query = projection_matrix(K, rvec, tvec)
    H       = plane_homography_z0(P_ref, P_query)

    if q_W != ref_W or q_H != ref_H:
        query_img = cv2.resize(query_img, (ref_W, ref_H))

    aligned = cv2.warpPerspective(query_img, H, (ref_W, ref_H))
    cv2.imwrite(output_path, aligned)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Align query images to a reference via DLT camera calibration"
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
                        help="Minimum RANSAC inlier keypoints required (must be ≥4). "
                             "All landmarks must be detected regardless of this value.")
    parser.add_argument("--ransac-threshold",    type=float, default=10.0,
                        help="RANSAC reprojection error threshold in pixels.")
    parser.add_argument("--no-subpixel",         action="store_true")
    args = parser.parse_args()

    if args.kp_weights is None and args.kp_weights_dir is None:
        parser.error("Provide --kp-weights (single model) or --kp-weights-dir (per-class).")
    if args.image is None and args.images_dir is None:
        parser.error("Provide --image or --images-dir.")

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Reference image ───────────────────────────────────────────────────
    ref_img = cv2.imread(args.reference)
    if ref_img is None:
        print(f"ERROR: cannot read reference image: {args.reference}")
        sys.exit(1)
    ref_H_px, ref_W_px = ref_img.shape[:2]
    image_size = (ref_W_px, ref_H_px)
    print(f"Reference image : {args.reference}  ({ref_W_px}×{ref_H_px})")

    # ── Reference keypoints ───────────────────────────────────────────────
    class_names, world_pts, ref_image_pts = load_reference_keypoints(
        args.reference_keypoints
    )
    print(f"Landmarks       : {class_names}")

    # UTM eastings/northings are in the hundreds-of-thousands to millions range,
    # which causes numerical instability in the DLT linear system.  Centre the
    # XY coordinates around their mean.  Z is left unchanged so that z=0 still
    # corresponds to the water surface.
    xy_origin  = world_pts[:, :2].mean(axis=0)
    world_pts  = world_pts.copy()
    world_pts[:, :2] -= xy_origin
    print(f"UTM XY origin   : ({xy_origin[0]:.2f}, {xy_origin[1]:.2f})  "
          f"(subtracted for numerical stability)")

    # ── Load models ───────────────────────────────────────────────────────
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

    # ── Pass 1: detect keypoints in all query images ──────────────────────
    print(f"\nPass 1: detecting keypoints in {len(image_paths)} image(s)…")
    all_detections: dict = {}   # img_path -> {class_name: [kp_x, kp_y]}

    for img_path in image_paths:
        results = run_pipeline(
            img_path, yolo_model, kp_models, cfg, device,
            subpixel=not args.no_subpixel,
        )
        all_detections[img_path] = {
            r.detection.class_name: [r.kp_x, r.kp_y]
            for r in results
        }

    # ── Calibration ───────────────────────────────────────────────────────
    # Reference image is always included first using its manually specified
    # 2D coords.  Query images are included only when all landmarks are detected
    # to ensure a consistent point configuration for calibration.
    print("\nCalibrating camera…")
    calib_pts = [ref_image_pts]
    for img_path in image_paths:
        dets = all_detections[img_path]
        if all(c in dets for c in class_names):
            pts = np.array([dets[c] for c in class_names], dtype=np.float32)
            # Scale keypoints to reference resolution if image size differs
            img_check  = cv2.imread(img_path)
            q_h, q_w   = img_check.shape[:2]
            if q_w != ref_W_px or q_h != ref_H_px:
                pts[:, 0] *= ref_W_px / q_w
                pts[:, 1] *= ref_H_px / q_h
            calib_pts.append(pts)

    print(
        f"  Images used for calibration: {len(calib_pts)} "
        f"(1 reference + {len(calib_pts) - 1} query)"
    )

    if len(calib_pts) < 3:
        print("ERROR: need at least 3 images with all landmarks detected for calibration.")
        sys.exit(1)

    K, rvecs, tvecs = calibrate_camera(calib_pts, world_pts, image_size)

    # Reference camera pose is rvecs[0] / tvecs[0] (first entry in calib_pts)
    P_ref = projection_matrix(K, rvecs[0], tvecs[0])

    # ── Pass 2: align each query image ────────────────────────────────────
    print(f"\nPass 2: aligning {len(image_paths)} image(s) → {out_dir}\n")

    n_success = 0
    for img_path in image_paths:
        stem     = Path(img_path).stem
        out_path = str(out_dir / f"{stem}_aligned.jpg")
        success  = align_image(
            query_path=img_path,
            class_names=class_names,
            world_pts=world_pts,
            detected_2d=all_detections[img_path],
            K=K,
            P_ref=P_ref,
            ref_size=(ref_W_px, ref_H_px),
            output_path=out_path,
            min_points=args.min_points,
            ransac_reproj_threshold=args.ransac_threshold,
        )
        if success:
            n_success += 1

    print(f"\nDone.  {n_success}/{len(image_paths)} images aligned successfully.")


if __name__ == "__main__":
    main()
