"""Geometric utilities for deep homography estimation.

Provides conversions between 4-point displacement parameterisation and
3x3 homography matrices, image warping, random perturbation generation,
and point transformation.
"""

import cv2
import numpy as np


def four_point_to_homography(four_point: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Convert 4-corner displacements to a 3x3 homography via DLT.

    Args:
        four_point: (8,) array of (dx, dy) displacements for the four corners
            ordered as: top-left, top-right, bottom-right, bottom-left.
        crop_size: (width, height) of the image patch.

    Returns:
        (3, 3) homography matrix H such that dst = H @ src.
    """
    w, h = crop_size
    # Canonical corner positions
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float64)

    # Displaced corners
    displacements = four_point.reshape(4, 2)
    dst_pts = src_pts + displacements

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def homography_to_four_point(H: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Convert a 3x3 homography to 4-corner displacements.

    Args:
        H: (3, 3) homography matrix.
        crop_size: (width, height) of the image patch.

    Returns:
        (8,) array of corner displacements.
    """
    w, h = crop_size
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float64)

    dst_pts = transform_points(src_pts, H)
    displacements = dst_pts - src_pts
    return displacements.flatten()


def scale_homography(
    H: np.ndarray,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    """Rescale a homography assuming both src and dst scale identically.

    Use scale_homography_asymmetric when source and destination images
    have different full resolutions.

    Args:
        H: (3, 3) homography computed at from_size resolution.
        from_size: (width, height) resolution H was computed at.
        to_size: (width, height) target resolution.

    Returns:
        (3, 3) homography valid at to_size resolution.
    """
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]

    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    S_inv = np.array([
        [1 / sx, 0, 0],
        [0, 1 / sy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return S @ H @ S_inv


def scale_homography_asymmetric(
    H: np.ndarray,
    working_size: tuple[int, int],
    src_full_size: tuple[int, int],
    dst_full_size: tuple[int, int],
) -> np.ndarray:
    """Rescale a homography when source and destination have different resolutions.

    H_working maps: working-res source coords -> working-res destination coords.
    H_full maps: full-res source coords -> full-res destination coords.

    H_full = S_dst @ H_working @ S_src_inv

    Args:
        H: (3, 3) homography at working resolution.
        working_size: (width, height) working resolution.
        src_full_size: (width, height) full resolution of source (target) image.
        dst_full_size: (width, height) full resolution of destination (reference) image.

    Returns:
        (3, 3) homography mapping full-res source to full-res destination.
    """
    # S_src_inv: full source -> working
    sx_src = working_size[0] / src_full_size[0]
    sy_src = working_size[1] / src_full_size[1]
    S_src_inv = np.array([
        [sx_src, 0, 0],
        [0, sy_src, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # S_dst: working -> full destination
    sx_dst = dst_full_size[0] / working_size[0]
    sy_dst = dst_full_size[1] / working_size[1]
    S_dst = np.array([
        [sx_dst, 0, 0],
        [0, sy_dst, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return S_dst @ H @ S_src_inv


def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int] = None,
) -> np.ndarray:
    """Warp an image by a homography.

    Args:
        image: Input image (H, W) or (H, W, C).
        H: (3, 3) homography matrix.
        output_size: (width, height) of output. Defaults to input size.

    Returns:
        Warped image.
    """
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)
    return cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)


def random_homography_perturbation(
    image_size: tuple[int, int],
    max_displacement: float = 32.0,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random 4-corner displacement and corresponding homography.

    Args:
        image_size: (width, height) of the working image.
        max_displacement: Maximum pixel displacement per corner.
        rng: NumPy random generator (for reproducibility).

    Returns:
        (four_point, H) where four_point is (8,) and H is (3, 3).
    """
    if rng is None:
        rng = np.random.default_rng()

    four_point = rng.uniform(-max_displacement, max_displacement, size=8).astype(np.float64)
    H = four_point_to_homography(four_point, image_size)
    return four_point, H


# ---------------------------------------------------------------------------
# Gauss-Newton (Inverse Compositional) homography refinement
# ---------------------------------------------------------------------------

def _homography_jacobian(px: np.ndarray, py: np.ndarray, w: int, h: int) -> np.ndarray:
    """Compute the Jacobian of the warp w.r.t. 8 homography parameters.

    For the SL(3) parameterisation with h33=1, the warp is:
        x' = (h1*x + h2*y + h3) / (h7*x + h8*y + 1)
        y' = (h4*x + h5*y + h6) / (h7*x + h8*y + 1)

    At the identity (h=[1,0,0, 0,1,0, 0,0]), the Jacobian dW/dp is:
        dx'/dh = [x, y, 1, 0, 0, 0, -x*x', -y*x'] / denom
        dy'/dh = [0, 0, 0, x, y, 1, -x*y', -y*y'] / denom

    At identity x'=x, y'=y, denom=1.

    Args:
        px: (N,) x-coordinates (normalised to [-1, 1]).
        py: (N,) y-coordinates (normalised to [-1, 1]).
        w: image width (for normalisation reference).
        h: image height (for normalisation reference).

    Returns:
        (N, 2, 8) Jacobian: J[i] is 2x8 for point i.
    """
    N = len(px)
    J = np.zeros((N, 2, 8), dtype=np.float64)

    # At identity warp
    J[:, 0, 0] = px      # dx'/dh1
    J[:, 0, 1] = py      # dx'/dh2
    J[:, 0, 2] = 1.0     # dx'/dh3
    J[:, 0, 6] = -px * px  # dx'/dh7
    J[:, 0, 7] = -py * px  # dx'/dh8

    J[:, 1, 3] = px      # dy'/dh4
    J[:, 1, 4] = py      # dy'/dh5
    J[:, 1, 5] = 1.0     # dy'/dh6
    J[:, 1, 6] = -px * py  # dy'/dh7
    J[:, 1, 7] = -py * py  # dy'/dh8

    return J


def gauss_newton_refine(
    H_init: np.ndarray,
    ref_gray: np.ndarray,
    target_gray: np.ndarray,
    mask: np.ndarray = None,
    num_iters: int = 15,
    convergence_thresh: float = 1e-4,
) -> np.ndarray:
    """Refine a homography using inverse compositional Gauss-Newton.

    Minimises the masked photometric error between the reference and the
    warped target by iteratively updating the homography.

    Uses the inverse compositional formulation: precompute the Jacobian
    and Hessian on the reference image once, then only recompute the
    error image each iteration.

    Args:
        H_init: (3, 3) initial homography (target -> reference).
        ref_gray: (H, W) float32/float64 reference grayscale in [0, 1].
        target_gray: (H, W) float32/float64 target grayscale in [0, 1].
        mask: (H, W) binary mask (non-zero = valid). None uses all pixels.
        num_iters: Maximum iterations.
        convergence_thresh: Stop if parameter update norm < this.

    Returns:
        (3, 3) refined homography.
    """
    h, w = ref_gray.shape[:2]
    ref = ref_gray.astype(np.float64)
    target = target_gray.astype(np.float64)

    if mask is None:
        mask_flat = np.ones(h * w, dtype=np.float64)
    else:
        mask_flat = mask.astype(np.float64).ravel()

    # Compute reference image gradients
    grad_x = cv2.Sobel(ref, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(ref, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    # Pixel coordinates (normalised is not needed — work in pixel coords)
    yy, xx = np.mgrid[0:h, 0:w]
    px = xx.ravel().astype(np.float64)
    py = yy.ravel().astype(np.float64)

    # Compute Jacobian of warp at identity: (N, 2, 8)
    J_warp = _homography_jacobian(px, py, w, h)

    # Steepest descent images: grad_I @ J_warp -> (N, 8)
    gx = grad_x.ravel()  # (N,)
    gy = grad_y.ravel()  # (N,)

    # SD[i, j] = gx[i] * J_warp[i, 0, j] + gy[i] * J_warp[i, 1, j]
    SD = gx[:, None] * J_warp[:, 0, :] + gy[:, None] * J_warp[:, 1, :]  # (N, 8)

    # Apply mask to steepest descent
    SD_masked = SD * mask_flat[:, None]  # (N, 8)

    # Precompute Hessian: H = SD^T @ SD  (8, 8)
    H_gn = SD_masked.T @ SD  # (8, 8)

    # Regularise for stability
    H_gn += np.eye(8) * 1e-6

    # Precompute inverse Hessian
    try:
        H_gn_inv = np.linalg.inv(H_gn)
    except np.linalg.LinAlgError:
        return H_init  # degenerate — return unchanged

    H_current = H_init.astype(np.float64).copy()
    ref_flat = ref.ravel()

    for iteration in range(num_iters):
        # Warp target by current H
        warped = cv2.warpPerspective(
            target, H_current, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        warped_flat = warped.ravel()

        # Error image: warped_target - reference
        error = (warped_flat - ref_flat) * mask_flat  # (N,)

        # Parameter update: dp = H_inv @ SD^T @ error
        rhs = SD_masked.T @ error  # (8,)
        dp = H_gn_inv @ rhs  # (8,)

        # Check convergence
        dp_norm = np.linalg.norm(dp)
        if dp_norm < convergence_thresh:
            break

        # Construct incremental warp from dp
        # dp parameterises H_delta as: [[1+p1, p2, p3], [p4, 1+p5, p6], [p7, p8, 1]]
        H_delta = np.array([
            [1.0 + dp[0], dp[1], dp[2]],
            [dp[3], 1.0 + dp[4], dp[5]],
            [dp[6], dp[7], 1.0],
        ], dtype=np.float64)

        # Inverse compositional update: H <- H @ H_delta_inv
        try:
            H_delta_inv = np.linalg.inv(H_delta)
        except np.linalg.LinAlgError:
            break
        H_current = H_current @ H_delta_inv

        # Normalise so H[2,2] = 1
        H_current = H_current / H_current[2, 2]

    return H_current


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a homography to 2D points.

    Args:
        points: (N, 2) array of (x, y) coordinates.
        H: (3, 3) homography matrix.

    Returns:
        (N, 2) array of transformed coordinates.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    # Convert to homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N, 3)

    # Apply homography
    dst_h = (H @ pts_h.T).T  # (N, 3)

    # Normalise
    dst = dst_h[:, :2] / dst_h[:, 2:3]
    return dst
