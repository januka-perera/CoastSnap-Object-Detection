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

With 5 point correspondences this gives 10 equations and 4 unknowns
(f + 3 rotation), translation is fixed to zero because the world is
pre-shifted so the camera centre is at the origin.  Solved by
scipy.optimize.least_squares (TRF, soft-L1 loss).

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
import traceback
sys.path.insert(0, str(Path(__file__).parent))

from predict import (
    load_config,
    load_yolo,
    load_keypoint_model,
    load_per_class_models,
    run_pipeline,
)
from tides import get_tide_height


# ---------------------------------------------------------------------------
# Reference keypoints
# ---------------------------------------------------------------------------

def load_reference_keypoints(path: str):
    """
    Load reference keypoints from JSON.

    The JSON may contain an optional top-level "camera_pos" key with the
    camera's UTM position [X, Y, Z].  All other keys are treated as GCPs
    with "image" and "world" sub-keys.

    Returns
    -------
    class_names    : ordered list of landmark class names
    world_pts      : (N, 3) float32 — 3D world coordinates in local metres
    ref_image_pts  : (N, 2) float32 — 2D pixel coords in the reference image
    camera_pos_utm : (3,) float64 UTM camera position, or None if not present
    """
    with open(path) as f:
        data = json.load(f)

    camera_pos_utm = None
    if "camera_pos" in data:
        camera_pos_utm = np.array(data["camera_pos"], dtype=np.float64)

    class_names   = []
    world_pts     = []
    ref_image_pts = []
    for cls, val in data.items():
        if not isinstance(val, dict) or "world" not in val or "image" not in val:
            continue
        class_names.append(cls)
        world_pts.append(val["world"])
        ref_image_pts.append(val["image"])

    return (
        class_names,
        np.array(world_pts,     dtype=np.float32),
        np.array(ref_image_pts, dtype=np.float32),
        camera_pos_utm,
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
    Estimate focal length f and rotation R for a single image, assuming the
    camera centre is at the world origin (world points already shifted by C).

    Model
    -----
    x ~ K [R | 0] X', where X' = X - C and C is the camera centre.
    Unknowns: f + rvec (4 DOF). Translation is fixed tvec = 0.

    Returns
    -------
    K, rvec, tvec(=0), inlier_mask
    """
    try:
        from scipy.optimize import least_squares as _lsq
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")

    W, H = image_size
    # Pixel-centre convention
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    pts_3d = world_pts.astype(np.float64).reshape(-1, 1, 3)
    pts_2d = image_pts.astype(np.float64).reshape(-1, 1, 2)

    # Resolution-scaled inlier threshold (px)
    thr = max(ransac_reproj_threshold, 0.006 * max(W, H))   # 0.6% of long edge
    thr = float(np.clip(thr, 6.0, 60.0))                   # clamp



    # ── Step 1: RANSAC bootstrap (gets a decent rvec) ────────────────────
    f_init = float(max(W, H))
    K_init = np.array([[f_init, 0., cx],
                       [0., f_init, cy],
                       [0., 0., 1.]], dtype=np.float64)

    ok, rvec0, tvec0, inliers0 = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_init, None,
        reprojectionError=thr,
        confidence=0.99,
        iterationsCount=2000,
    )
    if not ok or inliers0 is None or len(inliers0) < min_inliers:
        return None, None, None, None

    inlier_idx = inliers0.ravel()
    pts_3d_in = world_pts[inlier_idx].astype(np.float64)
    pts_2d_in = image_pts[inlier_idx].astype(np.float64)

    # ── Step 2: refine f + rvec with tvec fixed to 0 ─────────────────────
    tvec_fixed = np.zeros((3, 1), dtype=np.float64)

    def _residuals(x):
        f, rx, ry, rz = x
        K_ = np.array([[f, 0., cx],
                       [0., f, cy],
                       [0., 0., 1.]], dtype=np.float64)
        rvec = np.array([[rx], [ry], [rz]], dtype=np.float64)
        proj, _ = cv2.projectPoints(
            pts_3d_in.reshape(-1, 1, 3), rvec, tvec_fixed, K_, None
        )
        return (proj.reshape(-1, 2) - pts_2d_in).ravel()

    x0 = np.array([f_init, *rvec0.ravel()], dtype=np.float64)

    f_lo = 0.5 * max(W, H)
    f_hi = 5.0 * max(W, H)

    res = _lsq(
        _residuals, x0,
        method="trf",
        loss="soft_l1",
        f_scale=thr,
        bounds=([f_lo, -np.inf, -np.inf, -np.inf],
                [f_hi,  np.inf,  np.inf,  np.inf]),
    )

    f, rx, ry, rz = res.x
    if f <= 0:
        return None, None, None, None

    K    = np.array([[f, 0., cx],
                     [0., f, cy],
                     [0., 0., 1.]], dtype=np.float64)
    rvec = np.array([[rx], [ry], [rz]], dtype=np.float64)
    tvec = tvec_fixed

    # ── Step 3: final inlier mask over all N points ───────────────────────
    proj_all, _ = cv2.projectPoints(
        world_pts.astype(np.float64).reshape(-1, 1, 3), rvec, tvec, K, None
    )
    errors      = np.linalg.norm(proj_all.reshape(-1, 2) - image_pts, axis=1)
    inlier_mask = errors < thr

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
    z: float = 0.0,
) -> np.ndarray:
    """
    Warp an image to a north-up plan (bird's-eye) view via the z=h plane.

    Parameters
    ----------
    image : BGR image array (H, W, 3)
    P     : (3, 4) camera projection matrix in the **centroid-subtracted**
            world coordinate system (same system used when estimating the pose).
    xlim  : (Emin, Emax) in centroid-subtracted metres — output Easting extent.
    ylim  : (Nmin, Nmax) in centroid-subtracted metres — output Northing extent.
    dx    : ground-sampling distance in metres per output pixel.
    z     : elevation of the rectification plane in metres (default 0 = MSL).

    Returns
    -------
    plan : (ny, nx, 3) BGR plan-view image.
    """
    Emin, Emax = xlim
    Nmin, Nmax = ylim

    nx = int(round((Emax - Emin) / dx))
    ny = int(round((Nmax - Nmin) / dx))

    # Build world coordinate grid (north-up: row 0 = Nmax, row ny-1 = Nmin)
    # col c → X = Emin + c*dx
    # row r → Y = Nmax - r*dx
    C, R = np.meshgrid(np.arange(nx), np.arange(ny))
    X = Emin + C * dx
    Y = Nmax - R * dx

    # Project world [X, Y, z] → image homogeneous.
    # For a general z=h plane: P*[X,Y,h,1]^T = P[:,0]*X + P[:,1]*Y + (P[:,2]*h + P[:,3])*1
    B = np.column_stack([P[:, 0], P[:, 1], P[:, 2] * z + P[:, 3]])
    XY1 = np.stack([X, Y, np.ones_like(X)], axis=0).reshape(3, -1)  # (3, nx*ny)
    uvw = B @ XY1  # (3, nx*ny)

    # Perspective division then nearest-neighbour rounding (matches MATLAB toolbox)
    u = np.round(uvw[0] / uvw[2]).astype(int)
    v = np.round(uvw[1] / uvw[2]).astype(int)

    # Keep only pixels that fall within the image
    img_h, img_w = image.shape[:2]
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    # Copy pixels from source image into plan view
    plan = np.zeros((ny, nx, 3), dtype=np.uint8)
    plan[R.ravel()[valid], C.ravel()[valid]] = image[v[valid], u[valid]]
    return plan


def plane_homography_z0(
    P_ref: np.ndarray,
    P_query: np.ndarray,
    z: float = 0.0,
    cond_threshold: float = 1e8,
) -> np.ndarray:
    """
    Compute the homography induced by the z=h plane.

    For any point on z=h:  X_world = [X, Y, h, 1]^T
        p = P * X_world  =  P[:,0]*X + P[:,1]*Y + (P[:,2]*h + P[:,3])*1

    The homography mapping query image coords → reference image coords is:
        H = A_ref @ inv(A_query)  where A = [P[:,0], P[:,1], P[:,2]*h + P[:,3]]

    Uses np.linalg.solve for numerical stability instead of explicit inversion.
    Returns None if A_query is near-singular.
    """
    A = np.column_stack([P_ref  [:, 0], P_ref  [:, 1], P_ref  [:, 2] * z + P_ref  [:, 3]])
    B = np.column_stack([P_query[:, 0], P_query[:, 1], P_query[:, 2] * z + P_query[:, 3]])

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
    aligned_path: str,
    min_inliers: int = 4,
    ransac_reproj_threshold: float = 10.0,
    rectify: bool = False,
    rectify_path: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
    dx: float = None,
    tide_height: float = 0.0,
) -> bool:
    """
    Estimate per-image focal length and camera pose (4-DOF: f + R, tvec=0).

    Always saves the image aligned to the reference frame to `aligned_path`.
    When rectify=True, additionally saves a north-up plan-view image to
    `rectify_path`.

    Parameters
    ----------
    detected_2d  : dict mapping class_name → [kp_x, kp_y] in query pixels.
    P_ref        : (3, 4) reference camera projection matrix.
    ref_size     : (width, height) of the reference image — output warp size.
    min_inliers  : minimum RANSAC inliers required.
    aligned_path : output path for the reference-aligned image.
    rectify      : if True, also produce a plan-view image.
    rectify_path : output path for the plan-view image (required if rectify).
    xlim         : (Emin, Emax) in local metres (required if rectify).
    ylim         : (Nmin, Nmax) in local metres (required if rectify).
    dx           : ground-sampling distance m/px (required if rectify).
    tide_height  : z-plane elevation in the camera-centred local frame (metres).
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

    n_required = 4 # Min required for homography
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
        f"inliers={n_inliers}/{len(src_2d)}  "
        f"z'={tide_height:.2f} m"
    )

    # ── Build query projection matrix ────────────────────────────────────
    P_query = projection_matrix(K_query, rvec, tvec)

    # ── Align to reference frame (always) ────────────────────────────────
    # P_ref and P_query are both expressed in their own image coordinate
    # systems (different K), so the homography maps query pixels →
    # reference pixels without any manual resizing.
    H = plane_homography_z0(P_ref, P_query, z=tide_height)

    if H is None:
        print(f"  [SKIP] {stem}: plane homography is near-singular (degenerate pose)")
        return False

    ref_W, ref_H = ref_size
    aligned = cv2.warpPerspective(query_img, H, (ref_W, ref_H))
    cv2.imwrite(aligned_path, aligned)

    # ── Plan-view rectification (optional) ───────────────────────────────
    if rectify:
        plan = rectify_plan_view(query_img, P_query, xlim, ylim, dx, z=tide_height)
        cv2.imwrite(rectify_path, plan)
        nx, ny = plan.shape[1], plan.shape[0]
        print(f"  {stem}: plan view saved ({nx}×{ny} px, dx={dx} m/px)")

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
    parser.add_argument("--output-dir",          default="./aligned",
                        help="Output directory for aligned images (default: ./aligned).")
    parser.add_argument("--rectify-dir",         default="./rectified",
                        help="Output directory for plan-view images (default: ./rectified).")
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
                        help="Easting extent in local metres (required with --rectify).")
    parser.add_argument("--ylim",                type=float, nargs=2,
                        metavar=("N_MIN", "N_MAX"),
                        help="Northing extent in local metres (required with --rectify).")
    parser.add_argument("--dx",                  type=float, default=None,
                        help="Ground-sampling distance in metres/pixel "
                             "(required with --rectify).")

    # Tide height
    parser.add_argument("--utm-zone",            type=int, default=None,
                        help="MGA94/UTM zone number (e.g. 56 for Sydney). "
                             "When provided, tide height is fetched per image "
                             "and used as the rectification plane elevation. "
                             "Falls back to 0 m on any error.")
    parser.add_argument("--northern",            action="store_true",
                        help="Set if the site is in the northern hemisphere "
                             "(used for tide height lookup).")

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
    class_names, world_pts, ref_image_pts, camera_pos_utm = load_reference_keypoints(
        args.reference_keypoints
    )
    print(f"Landmarks       : {class_names}")

    # Shift the entire world so the camera centre is at the origin.
    # This fixes tvec=0 and reduces unknowns from 7 to 4 (f + rvec).
    if camera_pos_utm is None:
        print("ERROR: camera_pos must be provided for fixed-centre (4-DOF) mode.")
        sys.exit(1)

    origin = camera_pos_utm.astype(np.float64)
    print(
        f"World origin    : camera centre (UTM/MSL) = "
        f"({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})  "
        f"→ local camera-centred frame"
    )

    world_pts = world_pts.astype(np.float64)
    world_pts -= origin  # subtract full XYZ from every GCP

    camera_pos_local = np.zeros(3, dtype=np.float64)  # camera is at origin by construction
    print("Camera position : local (0.00, 0.00, 0.00)  [4-DOF fixed-centre mode]")

    if args.rectify:
        xlim_local = tuple(args.xlim)
        ylim_local = tuple(args.ylim)
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

    aligned_dir   = Path(args.output_dir)
    rectified_dir = Path(args.rectify_dir)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    if args.rectify:
        rectified_dir.mkdir(parents=True, exist_ok=True)

    # ── Detect keypoints then align each query image ──────────────────────
    print(
        f"\nProcessing {len(image_paths)} query image(s)\n"
        f"  aligned   → {aligned_dir}\n"
        + (f"  rectified → {rectified_dir}\n" if args.rectify else "")
    )

    n_success = 0
    for img_path in image_paths:
        stem = Path(img_path).stem

        # ── Tide height for this image ─────────────────────────────────────
        tide_height = 0.0
        if args.utm_zone is not None:
            try:
                tide_height = get_tide_height(
                    img_path,
                    keypoints_path=args.reference_keypoints,
                    utm_zone=args.utm_zone,
                    northern=args.northern,
                )
            except Exception as exc:
                traceback.print_exc()
                print(f"  {stem}: tide lookup failed ({exc}), defaulting to 0 m")

        # Convert MSL tide height to camera-centred local frame
        z_plane_local = tide_height - origin[2]
        print(f"  {stem}: tide MSL={tide_height:.2f} m  → plane z'={z_plane_local:.2f} m (camera-centred)")

        # Detect 2D landmark locations in this image
        results = run_pipeline(
            img_path, yolo_model, kp_models, cfg, device,
            subpixel=not args.no_subpixel,
        )
        detected_2d = {r.detection.class_name: [r.kp_x, r.kp_y] for r in results}

        aligned_path  = str(aligned_dir   / f"{stem}_aligned.jpg")
        rectify_path  = str(rectified_dir / f"{stem}_plan.jpg") if args.rectify else None
        success = align_image(
            query_path=img_path,
            class_names=class_names,
            world_pts=world_pts,
            detected_2d=detected_2d,
            P_ref=P_ref,
            ref_size=ref_size,
            aligned_path=aligned_path,
            min_inliers=args.min_inliers,
            ransac_reproj_threshold=args.ransac_threshold,
            rectify=args.rectify,
            rectify_path=rectify_path,
            xlim=xlim_local,
            ylim=ylim_local,
            dx=args.dx,
            tide_height=z_plane_local,
        )
        if success:
            n_success += 1

    print(f"\nDone.  {n_success}/{len(image_paths)} images aligned successfully.")


if __name__ == "__main__":
    main()