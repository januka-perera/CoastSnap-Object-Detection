"""
Heatmap utilities for keypoint prediction.

Functions
---------
generate_heatmap  : Create a 2-D Gaussian heatmap from a normalised (x, y) coordinate.
hard_argmax       : Non-differentiable peak extraction from a predicted heatmap.
soft_argmax       : Differentiable coordinate extraction via spatial softmax.
subpixel_refine   : Local quadratic fitting for sub-pixel peak refinement.
extract_coordinate: High-level wrapper that returns normalised (x, y) from a heatmap.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ground-truth heatmap generation
# ---------------------------------------------------------------------------

def generate_heatmap(
    x_norm: float,
    y_norm: float,
    heatmap_h: int,
    heatmap_w: int,
    sigma: float = 2.5,
) -> np.ndarray:
    """
    Generate a 2-D Gaussian heatmap for a single keypoint.

    Parameters
    ----------
    x_norm   : Normalised x coordinate in [0, 1]  (width axis).
    y_norm   : Normalised y coordinate in [0, 1]  (height axis).
    heatmap_h: Output heatmap height in pixels.
    heatmap_w: Output heatmap width  in pixels.
    sigma    : Gaussian standard deviation in heatmap pixels.

    Returns
    -------
    heatmap : float32 ndarray of shape (heatmap_h, heatmap_w), peak = 1.
    """
    kp_x = x_norm * (heatmap_w - 1)
    kp_y = y_norm * (heatmap_h - 1)

    xs = np.arange(heatmap_w, dtype=np.float32)
    ys = np.arange(heatmap_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)          # (H, W)

    heatmap = np.exp(
        -((grid_x - kp_x) ** 2 + (grid_y - kp_y) ** 2) / (2.0 * sigma ** 2)
    )
    peak = heatmap.max()
    if peak > 0.0:
        heatmap /= peak
    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Coordinate extraction from predicted heatmap
# ---------------------------------------------------------------------------

def hard_argmax(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Extract (x, y) coordinates using argmax.

    Parameters
    ----------
    heatmap : Tensor of shape (B, 1, H, W) or (B, H, W).

    Returns
    -------
    coords : Tensor of shape (B, 2) with (x_norm, y_norm) in [0, 1].
    """
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)      # (B, H, W)
    B, H, W = heatmap.shape

    flat = heatmap.view(B, -1)
    idx  = flat.argmax(dim=1)
    y_idx = idx // W
    x_idx = idx % W

    x_norm = x_idx.float() / (W - 1)
    y_norm = y_idx.float() / (H - 1)
    return torch.stack([x_norm, y_norm], dim=1)


def soft_argmax(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable coordinate extraction via spatial softmax.

    Parameters
    ----------
    heatmap     : Tensor of shape (B, 1, H, W) or (B, H, W).
    temperature : Softmax temperature — lower value sharpens the distribution.

    Returns
    -------
    coords : Tensor of shape (B, 2) with (x_norm, y_norm) in [0, 1].
    """
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)      # (B, H, W)
    B, H, W = heatmap.shape

    flat    = heatmap.view(B, -1) / temperature
    weights = F.softmax(flat, dim=1).view(B, H, W)

    xs = torch.linspace(0.0, 1.0, W, device=heatmap.device)
    ys = torch.linspace(0.0, 1.0, H, device=heatmap.device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")    # (H, W)

    x_norm = (weights * grid_x.unsqueeze(0)).sum(dim=[1, 2])
    y_norm = (weights * grid_y.unsqueeze(0)).sum(dim=[1, 2])
    return torch.stack([x_norm, y_norm], dim=1)


# ---------------------------------------------------------------------------
# Sub-pixel refinement
# ---------------------------------------------------------------------------

def subpixel_refine(
    heatmap: np.ndarray,
    x_idx: int,
    y_idx: int,
) -> tuple:
    """
    Apply 1-D quadratic fitting along each axis around the argmax peak.

    Parameters
    ----------
    heatmap : (H, W) float32 ndarray.
    x_idx   : Column index of the detected peak.
    y_idx   : Row    index of the detected peak.

    Returns
    -------
    (x_refined, y_refined) : Sub-pixel coordinates in pixel space.
    """
    H, W = heatmap.shape
    x_ref = float(x_idx)
    y_ref = float(y_idx)

    # X-axis
    if 0 < x_idx < W - 1:
        left   = float(heatmap[y_idx, x_idx - 1])
        centre = float(heatmap[y_idx, x_idx])
        right  = float(heatmap[y_idx, x_idx + 1])
        denom  = 2.0 * (2.0 * centre - left - right)
        if abs(denom) > 1e-6:
            x_ref = x_idx + (left - right) / denom

    # Y-axis
    if 0 < y_idx < H - 1:
        top    = float(heatmap[y_idx - 1, x_idx])
        centre = float(heatmap[y_idx,     x_idx])
        bottom = float(heatmap[y_idx + 1, x_idx])
        denom  = 2.0 * (2.0 * centre - top - bottom)
        if abs(denom) > 1e-6:
            y_ref = y_idx + (top - bottom) / denom

    return x_ref, y_ref


def extract_coordinate(
    heatmap: np.ndarray,
    heatmap_h: int,
    heatmap_w: int,
    subpixel: bool = True,
) -> tuple:
    """
    Extract a normalised (x_norm, y_norm) coordinate from a predicted heatmap.

    Parameters
    ----------
    heatmap   : (H, W) float32 ndarray.
    heatmap_h : Heatmap height.
    heatmap_w : Heatmap width.
    subpixel  : Apply quadratic sub-pixel refinement.

    Returns
    -------
    (x_norm, y_norm) in [0, 1].
    """
    flat_idx = int(np.argmax(heatmap))
    y_idx    = flat_idx // heatmap_w
    x_idx    = flat_idx  % heatmap_w

    if subpixel:
        x_px, y_px = subpixel_refine(heatmap, x_idx, y_idx)
    else:
        x_px, y_px = float(x_idx), float(y_idx)

    x_norm = x_px / (heatmap_w - 1)
    y_norm = y_px / (heatmap_h - 1)
    return x_norm, y_norm
