"""Gaussian heatmap generation and coordinate extraction utilities."""

import torch
import torch.nn.functional as F


def generate_gaussian_heatmap(
    cx: float,
    cy: float,
    width: int,
    height: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Generate a 2D Gaussian heatmap centred at (cx, cy).

    Args:
        cx: x-coordinate of the landmark in heatmap pixel space.
        cy: y-coordinate of the landmark in heatmap pixel space.
        width: Width of the output heatmap.
        height: Height of the output heatmap.
        sigma: Standard deviation of the Gaussian in pixels.

    Returns:
        Tensor of shape (height, width) with values in [0, 1].
    """
    xs = torch.arange(0, width, dtype=torch.float32)
    ys = torch.arange(0, height, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    heatmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


def generate_heatmaps_batch(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    width: int,
    height: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Generate ground-truth heatmaps for a batch of images.

    Args:
        keypoints: Normalised keypoint coords (B, N, 2) in [0, 1], (x, y) format.
        visibility: Boolean mask (B, N) indicating visible landmarks.
        width: Heatmap width.
        height: Heatmap height.
        sigma: Gaussian sigma in heatmap pixels.

    Returns:
        Heatmaps tensor of shape (B, N, height, width).
    """
    B, N, _ = keypoints.shape
    heatmaps = torch.zeros(B, N, height, width, dtype=torch.float32)

    for b in range(B):
        for n in range(N):
            if visibility[b, n]:
                cx = keypoints[b, n, 0] * (width - 1)
                cy = keypoints[b, n, 1] * (height - 1)
                heatmaps[b, n] = generate_gaussian_heatmap(cx, cy, width, height, sigma)

    return heatmaps


def argmax_coordinates(heatmaps: torch.Tensor) -> torch.Tensor:
    """Extract coordinates via argmax (integer accuracy).

    Args:
        heatmaps: (B, N, H, W) predicted heatmaps.

    Returns:
        Normalised coordinates (B, N, 2) in [0, 1], (x, y) format.
    """
    B, N, H, W = heatmaps.shape
    flat = heatmaps.view(B, N, -1)
    indices = flat.argmax(dim=-1)  # (B, N)

    y = (indices // W).float() / (H - 1)
    x = (indices % W).float() / (W - 1)

    return torch.stack([x, y], dim=-1)


def soft_argmax_coordinates(
    heatmaps: torch.Tensor,
    local_window_size: int = 0,
) -> torch.Tensor:
    """Extract coordinates via soft-argmax (sub-pixel accuracy).

    Args:
        heatmaps: (B, N, H, W) predicted heatmaps.
        local_window_size: If > 0, apply soft-argmax within a local window
            around the argmax peak. Must be odd. 0 means use full heatmap.

    Returns:
        Normalised coordinates (B, N, 2) in [0, 1], (x, y) format.
    """
    B, N, H, W = heatmaps.shape

    if local_window_size > 0:
        return _local_soft_argmax(heatmaps, local_window_size)

    # Create coordinate grids
    device = heatmaps.device
    xs = torch.arange(W, dtype=torch.float32, device=device) / (W - 1)
    ys = torch.arange(H, dtype=torch.float32, device=device) / (H - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    # Flatten spatial dimensions
    weights = heatmaps.view(B, N, -1)  # (B, N, H*W)
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    weights_norm = weights / weights_sum

    xx_flat = xx.reshape(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, H*W)
    yy_flat = yy.reshape(-1).unsqueeze(0).unsqueeze(0)

    x_coords = (weights_norm * xx_flat).sum(dim=-1)
    y_coords = (weights_norm * yy_flat).sum(dim=-1)

    return torch.stack([x_coords, y_coords], dim=-1)


def _local_soft_argmax(
    heatmaps: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Soft-argmax within a local window around the peak."""
    B, N, H, W = heatmaps.shape
    device = heatmaps.device
    half = window_size // 2

    # Get argmax positions
    flat = heatmaps.view(B, N, -1)
    indices = flat.argmax(dim=-1)
    peak_y = indices // W
    peak_x = indices % W

    coords = torch.zeros(B, N, 2, dtype=torch.float32, device=device)

    for b in range(B):
        for n in range(N):
            py, px = peak_y[b, n].item(), peak_x[b, n].item()
            x0 = max(0, px - half)
            x1 = min(W, px + half + 1)
            y0 = max(0, py - half)
            y1 = min(H, py + half + 1)

            patch = heatmaps[b, n, y0:y1, x0:x1]
            patch_sum = patch.sum().clamp(min=1e-8)

            local_xs = torch.arange(x0, x1, dtype=torch.float32, device=device)
            local_ys = torch.arange(y0, y1, dtype=torch.float32, device=device)
            local_yy, local_xx = torch.meshgrid(local_ys, local_xs, indexing="ij")

            coords[b, n, 0] = (patch * local_xx).sum() / patch_sum / (W - 1)
            coords[b, n, 1] = (patch * local_yy).sum() / patch_sum / (H - 1)

    return coords
