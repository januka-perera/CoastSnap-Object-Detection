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
