"""
Image alignment to a reference via per-image camera pose estimation.

Steps
-----
1. Load reference keypoints — manually specified 2D pixel coords in the
   reference image AND 3D world coords (local metres, z=up) for each landmark.
2. Estimate the reference camera pose (focal length + R + t) from the manually
   annotated 2D–3D correspondences using a nonlinear solver.
3. For each query image:
   a. Run YOLO + keypoint pipeline to detect 2D landmark locations.
   b. Estimate the query camera pose the same way.
   c. Compute the plane-induced homography for z=0 (water surface) between
      the query and reference cameras.
   d. Warp the query image with cv2.warpPerspective.

Why not homography?
-------------------
Homography requires scene points to be coplanar.  The landmarks here span
different elevations, so homography is not valid.  Instead we estimate the
full projection matrix P = K[R|t] per image and then derive the z=0 plane
homography analytically.

Per-image focal length
----------------------
Images may come from different smartphones with unknown intrinsics.  We
estimate focal length f independently for each image under the assumptions:
  - Zero skew
  - Square pixels: fx = fy = f
  - Principal point at image centre: cx = W/2, cy = H/2
  - No lens distortion

With 5 point correspondences this gives 10 equations and 7 unknowns
(f + 3 rotation + 3 translation), an overdetermined system solved by
Levenberg-Marquardt via scipy.optimize.least_squares.

reference_keypoints.json format
--------------------------------
    {
        "sign":       {"image": [1563.36, 1888.30], "world": [X, Y, Z]},
        "building-1": {"image": [2467.44,  652.89], "world": [X, Y, Z]},
        "building-2": {"image": [2768.24,  549.49], "world": [X, Y, Z]},
        "vent":       {"image": [1422.00, 2100.10], "world": [X, Y, Z]},
        "building-3": {"image": [u, v],             "world": [X, Y, Z]}
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
# Per-image camera pose estimation (f + R + t)
# ---------------------------------------------------------------------------

def estimate_camera_pose(
    image_pts: np.ndarray,
    world_pts: np.ndarray,
    image_size: tuple,
    ransac_reproj_threshold: float = 10.0,
    min_inliers: int = 4,
) -> tuple:
    """
    Jointly estimate focal length f and camera pose [R|t] for a single image.

    Assumptions
    -----------
    - Zero skew
    - Square pixels: fx = fy = f
    - Principal point at image centre: cx = W/2, cy = H/2
    - No lens distortion

    Strategy
    --------
    1. solvePnPRansac with a fixed initial-guess K (f = max(W, H)) to get a
       robust initial pose and inlier set.
    2. scipy.optimize.least_squares (LM) over inliers only to jointly refine
       f + rvec + tvec (7 parameters, 2xN_inliers equations).
    3. Compute final inlier mask with refined parameters over all N points.

    Parameters
    ----------
    image_pts              : (N, 2) float32 — 2D pixel coordinates
    world_pts              : (N, 3) float32 — 3D world coordinates
    image_size             : (width, height)
    ransac_reproj_threshold: reprojection error threshold for RANSAC and
                             final inlier counting (pixels)
    min_inliers            : minimum inliers required; returns None on failure

    Returns
    -------
    K           : (3, 3) float64 camera matrix with estimated f
    rvec        : (3, 1) float64 rotation vector
    tvec        : (3, 1) float64 translation vector
    inlier_mask : (N,) bool array — True for inlier points
    Returns (None, None, None, None) on failure.
    """
    try:
        from scipy.optimize import least_squares as _lsq
    except ImportError:
        raise ImportError(
            "scipy is required for per-image focal-length estimation. "
            "Install with:  pip install scipy"
        )

    W, H   = image_size
    cx, cy = W / 2.0, H / 2.0

    pts_3d = world_pts.astype(np.float64).reshape(-1, 1, 3)
    pts_2d = image_pts.astype(np.float64).reshape(-1, 1, 2)

    # ── Step 1: RANSAC with initial K to obtain a robust starting pose ────
    f_init = float(max(W, H))
    K_init = np.array(
        [[f_init, 0.,    cx],
         [0.,    f_init, cy],
         [0.,    0.,      1.]], dtype=np.float64,
    )

    ok, rvec0, tvec0, inliers0 = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_init, None,
        reprojectionError=ransac_reproj_threshold,
        confidence=0.99,
        iterationsCount=2000,
    )

    if not ok or inliers0 is None or len(inliers0) < min_inliers:
        return None, None, None, None

    inlier_idx = inliers0.ravel()

    # ── Step 2: LM refinement of f + pose jointly on inlier subset ───────
    pts_3d_in = world_pts[inlier_idx].astype(np.float64)
    pts_2d_in = image_pts[inlier_idx].astype(np.float64)

    def _residuals(params):
        f, rx, ry, rz, tx, ty, tz = params
        K_ = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]], dtype=np.float64)
        rv = np.array([rx, ry, rz], dtype=np.float64)
        tv = np.array([tx, ty, tz], dtype=np.float64)
        proj, _ = cv2.projectPoints(
            pts_3d_in.reshape(-1, 1, 3), rv, tv, K_, None
        )
        return (proj.reshape(-1, 2) - pts_2d_in).ravel()

    x0  = np.concatenate([[f_init], rvec0.ravel(), tvec0.ravel()])
    res = _lsq(_residuals, x0, method="lm")

    f, rx, ry, rz, tx, ty, tz = res.x

    if f <= 0:
        return None, None, None, None

    K    = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]], dtype=np.float64)
    rvec = np.array([[rx], [ry], [rz]], dtype=np.float64)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    # ── Step 3: final inlier mask over all N points ───────────────────────
    proj_all, _ = cv2.projectPoints(
        world_pts.astype(np.float64).reshape(-1, 1, 3), rvec, tvec, K, None
    )
    errors      = np.linalg.norm(proj_all.reshape(-1, 2) - image_pts, axis=1)
    inlier_mask = errors < ransac_reproj_threshold

    return K, rvec, tvec, inlier_mask


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


def rectify_plan_view(
    image: np.ndarray,
    P: np.ndarray,
    xlim: tuple,
    ylim: tuple,
    dx: float,
) -> np.ndarray:
    """
    Warp an image to a north-up plan (bird's-eye) view via the z=0 plane.

    Parameters
    ----------
    image : BGR image array (H, W, 3)
    P     : (3, 4) camera projection matrix in the **centroid-subtracted**
            world coordinate system (same system used when estimating the pose).
    xlim  : (Emin, Emax) in centroid-subtracted metres — output Easting extent.
    ylim  : (Nmin, Nmax) in centroid-subtracted metres — output Northing extent.
    dx    : ground-sampling distance in metres per output pixel.

    Returns
    -------
    plan : (ny, nx, 3) BGR plan-view image.
    """
    B = P[:, [0, 1, 3]]          # 3×3: maps [X, Y, 1]^T → image homogeneous

    Emin, Emax = xlim
    Nmin, Nmax = ylim

    nx = int(round((Emax - Emin) / dx))
    ny = int(round((Nmax - Nmin) / dx))

    # Map plan pixels → world [X, Y, 1]^T then → image coords
    # Plan pixel (c, r) → world (Emin + c*dx, Nmax - r*dx)
    M = np.array([
        [1 / dx,      0,  -Emin / dx],
        [0,      -1 / dx,  Nmax / dx],
        [0,           0,           1],
    ], dtype=np.float64)

    H = M @ np.linalg.inv(B)
    return cv2.warpPerspective(image, H, (nx, ny))


def plane_homography_z0(
    P_ref: np.ndarray,
    P_query: np.ndarray,
    cond_threshold: float = 1e8,
) -> np.ndarray:
    """
    Compute the homography induced by the z=0 plane.

    For any point on z=0:  X_world = [X, Y, 0, 1]^T
        p = P * X_world  =  P[:, [0,1,3]] * [X, Y, 1]^T

    The homography mapping query image coords → reference image coords is:
        H = P_ref[:, [0,1,3]] @ inv(P_query[:, [0,1,3]])

    Uses np.linalg.solve for numerical stability instead of explicit inversion.
    Returns None if P_query[:, [0,1,3]] is near-singular.
    """
    A = P_ref  [:, [0, 1, 3]]
    B = P_query[:, [0, 1, 3]]

    if np.linalg.cond(B) > cond_threshold:
        return None

    # solve B^T H^T = A^T  →  H = (B^T \ A^T)^T
    H = np.linalg.solve(B.T, A.T).T
    H /= H[2, 2]
    return H


# ---------------------------------------------------------------------------
# Per-image alignment
# ---------------------------------------------------------------------------

def align_image(
    query_path: str,
    class_names: list,
    world_pts: np.ndarray,
    detected_2d: dict,
    P_ref: np.ndarray,
    ref_size: tuple,
    output_path: str,
    min_inliers: int = 4,
    ransac_reproj_threshold: float = 10.0,
    rectify: bool = False,
    xlim: tuple = None,
    ylim: tuple = None,
    dx: float = None,
) -> bool:
    """
    Estimate per-image focal length and camera pose, then either:
      • align to the reference frame (default), or
      • produce a north-up plan-view rectification (rectify=True).

    Parameters
    ----------
    detected_2d  : dict mapping class_name → [kp_x, kp_y] in query image pixels.
    P_ref        : (3, 4) reference camera projection matrix.
    ref_size     : (width, height) of the reference image — output warp size.
    min_inliers  : minimum RANSAC inliers required (in addition to all landmarks
                   being detected).
    rectify      : if True, produce a plan-view image instead of aligning to ref.
    xlim         : (Emin, Emax) in centroid-subtracted metres (required if rectify).
    ylim         : (Nmin, Nmax) in centroid-subtracted metres (required if rectify).
    dx           : ground-sampling distance m/px (required if rectify).
    """
    stem = Path(query_path).name

    # ── Require all landmarks to be detected ──────────────────────────────
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
        missing = [c for c in class_names if c not in detected_2d]
        print(
            f"  [SKIP] {stem}: {len(src_2d)}/{n_required} landmarks detected "
            f"— missing {missing}"
        )
        return False

    # ── Load query image ──────────────────────────────────────────────────
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"  [SKIP] {stem}: cannot read image")
        return False

    q_H, q_W = query_img.shape[:2]

    src_2d = np.array(src_2d, dtype=np.float32)
    src_3d = np.array(src_3d, dtype=np.float32)

    # ── Estimate per-image focal length and pose ──────────────────────────
    # Keypoints are in query image pixel space; K is estimated for that space.
    K_query, rvec, tvec, inlier_mask = estimate_camera_pose(
        src_2d, src_3d, (q_W, q_H),
        ransac_reproj_threshold=ransac_reproj_threshold,
        min_inliers=min_inliers,
    )

    if K_query is None:
        print(f"  [SKIP] {stem}: pose estimation failed (too few RANSAC inliers)")
        return False

    n_inliers = int(inlier_mask.sum())
    if n_inliers < min_inliers:
        print(
            f"  [SKIP] {stem}: only {n_inliers}/{len(src_2d)} inliers "
            f"after refinement"
        )
        return False

    print(
        f"  {stem}: f={K_query[0, 0]:.1f} px  "
        f"inliers={n_inliers}/{len(src_2d)}"
    )

    # ── Build query projection matrix ────────────────────────────────────
    P_query = projection_matrix(K_query, rvec, tvec)

    if rectify:
        # ── Plan-view rectification ───────────────────────────────────────
        plan = rectify_plan_view(query_img, P_query, xlim, ylim, dx)
        cv2.imwrite(output_path, plan)
        nx, ny = plan.shape[1], plan.shape[0]
        print(f"  {stem}: plan view saved ({nx}×{ny} px, dx={dx} m/px)")
    else:
        # ── Align to reference frame ──────────────────────────────────────
        # P_ref and P_query are both expressed in their own image coordinate
        # systems (different K), so the homography maps query pixels →
        # reference pixels without any manual resizing.
        H = plane_homography_z0(P_ref, P_query)

        if H is None:
            print(f"  [SKIP] {stem}: plane homography is near-singular (degenerate pose)")
            return False

        ref_W, ref_H = ref_size
        aligned = cv2.warpPerspective(query_img, H, (ref_W, ref_H))
        cv2.imwrite(output_path, aligned)

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Align query images to a reference via per-image camera pose estimation"
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
    parser.add_argument("--min-inliers",         type=int, default=4,
                        help="Minimum RANSAC inliers required per image (≥4). "
                             "All landmarks must also be detected.")
    parser.add_argument("--ransac-threshold",    type=float, default=10.0,
                        help="RANSAC reprojection error threshold in pixels.")
    parser.add_argument("--no-subpixel",         action="store_true")

    # Plan-view rectification
    parser.add_argument("--rectify",             action="store_true",
                        help="Produce a north-up plan-view image instead of "
                             "aligning to the reference frame.")
    parser.add_argument("--xlim",                type=float, nargs=2,
                        metavar=("E_MIN", "E_MAX"),
                        help="Easting extent in UTM metres (required with --rectify).")
    parser.add_argument("--ylim",                type=float, nargs=2,
                        metavar=("N_MIN", "N_MAX"),
                        help="Northing extent in UTM metres (required with --rectify).")
    parser.add_argument("--dx",                  type=float, default=None,
                        help="Ground-sampling distance in metres/pixel "
                             "(required with --rectify).")

    args = parser.parse_args()

    if args.rectify:
        if args.xlim is None or args.ylim is None or args.dx is None:
            parser.error("--rectify requires --xlim, --ylim, and --dx.")

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
    ref_size = (ref_W_px, ref_H_px)
    print(f"Reference image : {args.reference}  ({ref_W_px}×{ref_H_px})")

    # ── Reference keypoints ───────────────────────────────────────────────
    class_names, world_pts, ref_image_pts = load_reference_keypoints(
        args.reference_keypoints
    )
    print(f"Landmarks       : {class_names}")

    # UTM eastings/northings are in the hundreds-of-thousands range.
    # Subtract the XY centroid for numerical stability in the nonlinear solve.
    # Z is left unchanged so z=0 still corresponds to the water surface.
    xy_origin = world_pts[:, :2].mean(axis=0)
    world_pts = world_pts.copy()
    world_pts[:, :2] -= xy_origin
    print(
        f"UTM XY origin   : ({xy_origin[0]:.2f}, {xy_origin[1]:.2f})  "
        f"(subtracted for numerical stability)"
    )

    # Convert user-supplied UTM xlim/ylim to centroid-subtracted coordinates
    # so they are consistent with the coordinate system used for pose estimation.
    if args.rectify:
        xlim_local = (args.xlim[0] - xy_origin[0], args.xlim[1] - xy_origin[0])
        ylim_local = (args.ylim[0] - xy_origin[1], args.ylim[1] - xy_origin[1])
        print(
            f"Plan-view extents (local): "
            f"X=[{xlim_local[0]:.1f}, {xlim_local[1]:.1f}]  "
            f"Y=[{ylim_local[0]:.1f}, {ylim_local[1]:.1f}]  "
            f"dx={args.dx} m/px"
        )
    else:
        xlim_local = ylim_local = None

    # ── Estimate reference camera pose ────────────────────────────────────
    print("\nEstimating reference camera pose…")
    K_ref, rvec_ref, tvec_ref, inliers_ref = estimate_camera_pose(
        ref_image_pts, world_pts, ref_size,
        ransac_reproj_threshold=args.ransac_threshold,
        min_inliers=4,
    )
    if K_ref is None:
        print("ERROR: could not estimate reference camera pose — too few inliers.")
        sys.exit(1)
    print(
        f"  f={K_ref[0, 0]:.1f} px  "
        f"inliers={inliers_ref.sum()}/{len(ref_image_pts)}"
    )
    P_ref = projection_matrix(K_ref, rvec_ref, tvec_ref)

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

    # ── Detect keypoints then align each query image ──────────────────────
    print(f"\nProcessing {len(image_paths)} query image(s) → {out_dir}\n")

    n_success = 0
    for img_path in image_paths:
        stem = Path(img_path).stem

        # Detect 2D landmark locations in this image
        results = run_pipeline(
            img_path, yolo_model, kp_models, cfg, device,
            subpixel=not args.no_subpixel,
        )
        detected_2d = {r.detection.class_name: [r.kp_x, r.kp_y] for r in results}

        suffix   = "_plan" if args.rectify else "_aligned"
        out_path = str(out_dir / f"{stem}{suffix}.jpg")
        success  = align_image(
            query_path=img_path,
            class_names=class_names,
            world_pts=world_pts,
            detected_2d=detected_2d,
            P_ref=P_ref,
            ref_size=ref_size,
            output_path=out_path,
            min_inliers=args.min_inliers,
            ransac_reproj_threshold=args.ransac_threshold,
            rectify=args.rectify,
            xlim=xlim_local,
            ylim=ylim_local,
            dx=args.dx,
        )
        if success:
            n_success += 1

    print(f"\nDone.  {n_success}/{len(image_paths)} images aligned successfully.")


if __name__ == "__main__":
    main()