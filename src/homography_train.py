"""Training loop for deep homography estimation.

Stage 1 (Synthetic): Train on unlimited synthetic pairs with L1 loss
    on 4-point displacements.
Stage 2 (Real fine-tuning, optional): Self-supervised alignment using
    masked photometric loss on real image pairs.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from .homography_model import HomographyNet
from .homography_dataset import SyntheticHomographyDataset, RealPairDataset
from .homography_real_dataset import RealGCPHomographyDataset, split_indices
from .homography_utils import four_point_to_homography, warp_image


def _four_point_to_homography_torch(four_point: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Differentiable 4-point to 3x3 homography (batched).

    Uses direct linear solve (Ah = b with h33=1) instead of SVD for
    numerical stability on GPU.

    Args:
        four_point: (B, 8) corner displacements.
        size: (width, height) of the image.

    Returns:
        (B, 3, 3) homography matrices.
    """
    B = four_point.shape[0]
    w, h = size
    device = four_point.device
    dtype = torch.float32

    four_point = four_point.float()

    # Canonical corners
    src = torch.tensor([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype=dtype, device=device)

    dst = src.unsqueeze(0) + four_point.view(B, 4, 2)  # (B, 4, 2)
    src = src.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 2)

    # Build 8x8 system Ah = b where h = [h11,h12,h13,h21,h22,h23,h31,h32] and h33=1
    # For each point (sx,sy) -> (dx,dy):
    #   sx*h11 + sy*h12 + h13 - sx*dx*h31 - sy*dx*h32 = dx
    #   sx*h21 + sy*h22 + h23 - sx*dy*h31 - sy*dy*h32 = dy
    A = torch.zeros(B, 8, 8, dtype=dtype, device=device)
    b = torch.zeros(B, 8, dtype=dtype, device=device)

    for i in range(4):
        sx = src[:, i, 0]
        sy = src[:, i, 1]
        dx = dst[:, i, 0]
        dy = dst[:, i, 1]

        # Row 2*i: equation for x'
        A[:, 2 * i, 0] = sx
        A[:, 2 * i, 1] = sy
        A[:, 2 * i, 2] = 1
        A[:, 2 * i, 6] = -sx * dx
        A[:, 2 * i, 7] = -sy * dx
        b[:, 2 * i] = dx

        # Row 2*i+1: equation for y'
        A[:, 2 * i + 1, 3] = sx
        A[:, 2 * i + 1, 4] = sy
        A[:, 2 * i + 1, 5] = 1
        A[:, 2 * i + 1, 6] = -sx * dy
        A[:, 2 * i + 1, 7] = -sy * dy
        b[:, 2 * i + 1] = dy

    h = torch.linalg.solve(A, b)  # (B, 8)

    H = torch.ones(B, 9, dtype=dtype, device=device)
    H[:, :8] = h
    return H.view(B, 3, 3)


def _differentiable_warp(image: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Differentiable image warping using grid_sample.

    Args:
        image: (B, 1, H, W) input images.
        H: (B, 3, 3) homography matrices.

    Returns:
        (B, 1, H, W) warped images.
    """
    B, _, Himg, Wimg = image.shape
    device = image.device

    # Create normalised grid [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, Himg, device=device),
        torch.linspace(-1, 1, Wimg, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    grid = torch.stack([xx, yy, ones], dim=-1).view(-1, 3)  # (H*W, 3)

    # Apply H inverse to get source coordinates
    H_inv = torch.linalg.inv(H)  # (B, 3, 3)
    grid_src = torch.bmm(H_inv, grid.unsqueeze(0).expand(B, -1, -1).permute(0, 2, 1))  # (B, 3, H*W)
    grid_src = grid_src.permute(0, 2, 1)  # (B, H*W, 3)
    grid_src = grid_src[..., :2] / grid_src[..., 2:3]  # normalise

    grid_src = grid_src.view(B, Himg, Wimg, 2)
    return torch.nn.functional.grid_sample(image, grid_src, mode="bilinear",
                                           padding_mode="border", align_corners=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """L1 loss on 4-point displacements, optionally weighted by mask.

    For Stage 1, mask is not used (loss is on displacement vectors, not pixels).
    This is a simple L1 on the 8-dim output.

    Args:
        pred: (B, 8) predicted displacements.
        target: (B, 8) ground-truth displacements.
        mask: Unused for displacement loss (kept for API consistency).

    Returns:
        Scalar loss.
    """
    return torch.mean(torch.abs(pred - target))


def _masked_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute masked SSIM between two single-channel images.

    Args:
        x: (B, 1, H, W) image.
        y: (B, 1, H, W) image.
        mask: (B, H, W) static region mask.
        window_size: Size of the Gaussian averaging window.

    Returns:
        Scalar SSIM loss (1 - SSIM, so lower is better).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    # Uniform averaging kernel
    kernel = torch.ones(1, 1, window_size, window_size, device=x.device, dtype=x.dtype)
    kernel = kernel / (window_size * window_size)

    mu_x = torch.nn.functional.conv2d(x, kernel, padding=pad)
    mu_y = torch.nn.functional.conv2d(y, kernel, padding=pad)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = torch.nn.functional.conv2d(x * x, kernel, padding=pad) - mu_x_sq
    sigma_y_sq = torch.nn.functional.conv2d(y * y, kernel, padding=pad) - mu_y_sq
    sigma_xy = torch.nn.functional.conv2d(x * y, kernel, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    ssim_map = ssim_map.squeeze(1)  # (B, H, W)

    # Apply mask
    masked_ssim = ssim_map * mask
    num_pixels = mask.sum().clamp(min=1.0)
    return 1.0 - masked_ssim.sum() / num_pixels


def masked_photometric_loss(
    warped: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    ssim_weight: float = 0.5,
) -> torch.Tensor:
    """Masked L1 + SSIM photometric loss for self-supervised fine-tuning.

    Combines L1 (pixel-level accuracy) with SSIM (structural similarity),
    weighted by ssim_weight. SSIM is less sensitive to global brightness
    changes and focuses on edges and patterns.

    Args:
        warped: (B, 1, H, W) target warped to reference frame.
        reference: (B, 1, H, W) reference image.
        mask: (B, H, W) static region mask (1=static, 0=dynamic).
        ssim_weight: Weight for SSIM component (0-1). L1 weight = 1 - ssim_weight.

    Returns:
        Scalar loss.
    """
    # L1 component
    diff = torch.abs(warped.squeeze(1) - reference.squeeze(1))  # (B, H, W)
    masked_diff = diff * mask
    num_pixels = mask.sum().clamp(min=1.0)
    l1_loss = masked_diff.sum() / num_pixels

    # SSIM component
    ssim_loss = _masked_ssim(warped, reference, mask)

    return (1.0 - ssim_weight) * l1_loss + ssim_weight * ssim_loss


def compute_mace(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Average Corner Error in pixels.

    Args:
        pred: (B, 8) predicted displacements.
        target: (B, 8) ground-truth displacements.

    Returns:
        MACE in pixels (scalar).
    """
    diff = (pred - target).view(-1, 4, 2)  # (B, 4, 2)
    corner_errors = torch.norm(diff, dim=2)  # (B, 4)
    return corner_errors.mean().item()


def train_stage1(config_path: str = "configs/config.yaml"):
    """Stage 1: Train on synthetic data with L1 displacement loss."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hcfg = cfg["homography"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    ckpt_dir = Path(hcfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"]) / "homography"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    working_size = (hcfg["working_width"], hcfg["working_height"])
    syn_cfg = hcfg["synthetic"]

    train_dataset = SyntheticHomographyDataset(
        reference_image_path=hcfg["reference_image"],
        working_size=working_size,
        max_displacement=syn_cfg["max_displacement"],
        samples_per_epoch=syn_cfg["samples_per_epoch"],
        mask_path=hcfg.get("mask_path"),
        augment=True,
        seed=42,
    )

    val_dataset = SyntheticHomographyDataset(
        reference_image_path=hcfg["reference_image"],
        working_size=working_size,
        max_displacement=syn_cfg["max_displacement"],
        samples_per_epoch=500,
        mask_path=hcfg.get("mask_path"),
        augment=False,
        seed=123,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=syn_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=syn_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    model_cfg = hcfg["model"]
    model = HomographyNet(
        backbone=model_cfg["backbone"],
        pretrained=True,
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)

    # Training config
    epochs = syn_cfg["epochs"]
    freeze_epochs = 5
    encoder_lr = 1e-5
    head_lr = 1e-4
    use_amp = device.type == "cuda"

    # Start with frozen encoder
    model.freeze_encoder()
    optimizer = torch.optim.Adam(
        model.get_param_groups(encoder_lr=0.0, head_lr=head_lr),
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 25
    history = {"train_loss": [], "val_loss": [], "val_mace": []}

    for epoch in range(1, epochs + 1):
        # Unfreeze encoder after warmup
        if epoch == freeze_epochs + 1:
            logger.info("Unfreezing encoder")
            model.unfreeze_encoder()
            optimizer = torch.optim.Adam(
                model.get_param_groups(encoder_lr=encoder_lr, head_lr=head_lr),
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10,
            )

        # Training
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            gt_four_point = batch["four_point"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(inputs)
                loss = masked_l1_loss(pred, gt_four_point)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_mace_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                gt_four_point = batch["four_point"].to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(inputs)
                    loss = masked_l1_loss(pred, gt_four_point)

                mace = compute_mace(pred, gt_four_point)

                val_loss_sum += loss.item()
                val_mace_sum += mace
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_mace = val_mace_sum / max(val_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mace"].append(avg_val_mace)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[-1]["lr"]

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val MACE: {avg_val_mace:.2f}px | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_mace": avg_val_mace,
                "config": cfg,
            }, ckpt_dir / "best_homography.pth")
            logger.info(f"  -> Saved best model (val_loss={avg_val_loss:.4f}, MACE={avg_val_mace:.2f}px)")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Save history
    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f)

    logger.info("Stage 1 training complete.")
    return model


def train_stage2(config_path: str = "configs/config.yaml"):
    """Stage 2: Self-supervised fine-tuning on real image pairs.

    Uses masked photometric loss (L1 within static regions) to fine-tune
    the homography network on actual image pairs without GT homography.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hcfg = cfg["homography"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ckpt_dir = Path(hcfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"]) / "homography"

    working_size = (hcfg["working_width"], hcfg["working_height"])

    # Load Stage 1 model
    checkpoint_path = ckpt_dir / "best_homography.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = hcfg["model"]
    model = HomographyNet(
        backbone=model_cfg["backbone"],
        pretrained=False,
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded Stage 1 model from {checkpoint_path}")

    # Build real-pair dataset
    real_dataset = RealPairDataset(
        images_dir=cfg["data"]["images_dir"],
        reference_image_path=hcfg["reference_image"],
        working_size=working_size,
        mask_path=hcfg.get("mask_path"),
    )

    real_loader = DataLoader(
        real_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Lower learning rates for fine-tuning
    optimizer = torch.optim.Adam(
        model.get_param_groups(encoder_lr=1e-6, head_lr=5e-5),
        weight_decay=1e-4,
    )

    # Disable AMP for Stage 2 — half-precision causes NaN in matrix inverse/solve
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Cache reference tensor
    ref_np = real_dataset.reference  # (H, W) float32
    ref_tensor = torch.from_numpy(ref_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    epochs = 50
    best_loss = float("inf")
    history = {"train_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in real_loader:
            inputs = batch["input"].to(device)
            mask = batch["mask"].to(device)
            B = inputs.shape[0]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_four_point = model(inputs)

                # Differentiable warp: target -> reference frame
                H_batch = _four_point_to_homography_torch(pred_four_point, working_size)
                target_images = inputs[:, 1:2, :, :]  # (B, 1, H, W)
                warped_tensor = _differentiable_warp(target_images, H_batch)

                ref_batch = ref_tensor.expand(B, -1, -1, -1)
                loss = masked_photometric_loss(warped_tensor, ref_batch, mask, ssim_weight=0.2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if np.isnan(epoch_loss):
                logger.warning("NaN loss detected, stopping training")
                break
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_loss)

        logger.info(f"Stage 2 Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_loss,
                "stage": 2,
                "config": cfg,
            }, ckpt_dir / "best_homography_finetuned.pth")

    with open(log_dir / "history_stage2.json", "w") as f:
        json.dump(history, f)

    logger.info("Stage 2 fine-tuning complete.")
    return model


# ---------------------------------------------------------------------------
# GCP reprojection loss
# ---------------------------------------------------------------------------

def gcp_reprojection_loss(
    pred_four_point: torch.Tensor,
    target_gcps: torch.Tensor,
    ref_gcps: torch.Tensor,
    working_size: tuple[int, int],
) -> torch.Tensor:
    """GCP reprojection loss — directly optimises the evaluation metric.

    Converts predicted 4-point displacements to a homography, transforms
    target GCPs, and computes L1 error against reference GCPs.

    Args:
        pred_four_point: (B, 8) predicted corner displacements.
        target_gcps: (B, N, 2) target GCPs at working resolution.
        ref_gcps: (B, N, 2) reference GCPs at working resolution.
        working_size: (width, height) of the working image.

    Returns:
        Scalar L1 loss.
    """
    H_pred = _four_point_to_homography_torch(pred_four_point, working_size)

    B, N, _ = target_gcps.shape
    ones = torch.ones(B, N, 1, device=target_gcps.device, dtype=target_gcps.dtype)
    pts_h = torch.cat([target_gcps, ones], dim=-1)  # (B, N, 3)

    # Apply H: (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    transformed_h = torch.bmm(H_pred, pts_h.permute(0, 2, 1))
    # De-homogenise: (B, 2, N) / (B, 1, N)
    transformed = transformed_h[:, :2, :] / transformed_h[:, 2:3, :].clamp(min=1e-8)
    transformed = transformed.permute(0, 2, 1)  # (B, N, 2)

    return torch.mean(torch.abs(transformed - ref_gcps))


def compute_gcp_error(
    pred_four_point: torch.Tensor,
    target_gcps: torch.Tensor,
    ref_gcps: torch.Tensor,
    working_size: tuple[int, int],
) -> float:
    """Compute mean Euclidean GCP error in pixels (for validation).

    Same as gcp_reprojection_loss but returns Euclidean distance
    instead of L1, matching the evaluation metric.
    """
    H_pred = _four_point_to_homography_torch(pred_four_point, working_size)

    B, N, _ = target_gcps.shape
    ones = torch.ones(B, N, 1, device=target_gcps.device, dtype=target_gcps.dtype)
    pts_h = torch.cat([target_gcps, ones], dim=-1)

    transformed_h = torch.bmm(H_pred, pts_h.permute(0, 2, 1))
    transformed = transformed_h[:, :2, :] / transformed_h[:, 2:3, :].clamp(min=1e-8)
    transformed = transformed.permute(0, 2, 1)

    errors = torch.norm(transformed - ref_gcps, dim=-1)  # (B, N)
    return errors.mean().item()


# ---------------------------------------------------------------------------
# Stage 3: Real-supervised training with GCP reprojection loss
# ---------------------------------------------------------------------------

def train_real(config_path: str = "configs/config.yaml", from_checkpoint: str = None):
    """Train on real image pairs with GCP reprojection loss.

    Uses annotated GCP correspondences to compute ground-truth homographies,
    then trains the model to minimise reprojection error on those GCPs.

    Can warm-start from a synthetic or fine-tuned checkpoint.

    Args:
        config_path: Path to config yaml.
        from_checkpoint: Optional path to checkpoint to initialise from.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hcfg = cfg["homography"]
    rcfg = hcfg.get("real_supervised", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Output directories
    ckpt_dir = Path(hcfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"]) / "homography"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    working_size = (hcfg["working_width"], hcfg["working_height"])

    # Reference .mat path
    ref_mat_path = hcfg.get(
        "reference_mat", cfg.get("refinement", {}).get("reference_mat"),
    )
    if ref_mat_path is None:
        raise ValueError("reference_mat not found in config")

    # Hyperparameters (with defaults)
    batch_size = rcfg.get("batch_size", 4)
    epochs = rcfg.get("epochs", 100)
    encoder_lr = rcfg.get("encoder_lr", 1e-5)
    head_lr = rcfg.get("head_lr", 1e-4)
    freeze_epochs = rcfg.get("freeze_encoder_epochs", 5)
    patience = rcfg.get("early_stopping_patience", 20)
    train_ratio = rcfg.get("train_ratio", 0.70)
    val_ratio = rcfg.get("val_ratio", 0.15)
    split_seed = rcfg.get("seed", 42)
    gcp_dir = rcfg.get("gcp_dir", "data/gcp")
    displacement_weight = rcfg.get("displacement_weight", 0.0)

    # Discover all valid samples (to compute split)
    full_dataset = RealGCPHomographyDataset(
        gcp_dir=gcp_dir,
        images_dir=cfg["data"]["images_dir"],
        reference_image_path=hcfg["reference_image"],
        reference_mat_path=ref_mat_path,
        working_size=working_size,
        mask_path=hcfg.get("mask_path"),
        sample_indices=None,
        augment=False,
    )
    n_total = len(full_dataset)
    logger.info(f"Total valid samples: {n_total}")

    # Split indices
    train_idx, val_idx, test_idx = split_indices(
        n_total, train_ratio, val_ratio, split_seed,
    )
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Save split for reproducibility
    split_info = {"train": train_idx, "val": val_idx, "test": test_idx}
    with open(log_dir / "real_data_split.json", "w") as f:
        json.dump(split_info, f)

    # Create train/val datasets
    train_dataset = RealGCPHomographyDataset(
        gcp_dir=gcp_dir,
        images_dir=cfg["data"]["images_dir"],
        reference_image_path=hcfg["reference_image"],
        reference_mat_path=ref_mat_path,
        working_size=working_size,
        mask_path=hcfg.get("mask_path"),
        sample_indices=train_idx,
        augment=True,
        seed=42,
    )

    val_dataset = RealGCPHomographyDataset(
        gcp_dir=gcp_dir,
        images_dir=cfg["data"]["images_dir"],
        reference_image_path=hcfg["reference_image"],
        reference_mat_path=ref_mat_path,
        working_size=working_size,
        mask_path=hcfg.get("mask_path"),
        sample_indices=val_idx,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Build model
    model_cfg = hcfg["model"]
    model = HomographyNet(
        backbone=model_cfg["backbone"],
        pretrained=True,
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)

    # Optionally warm-start from checkpoint
    ckpt_path = from_checkpoint or rcfg.get("from_checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Warm-started from {ckpt_path}")
    else:
        logger.info("Training from scratch (no checkpoint)")

    # Disable AMP — half-precision can cause NaN in linalg.solve
    use_amp = False

    # Start with frozen encoder
    model.freeze_encoder()
    optimizer = torch.optim.Adam(
        model.get_param_groups(encoder_lr=0.0, head_lr=head_lr),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    best_val_error = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_gcp_error": []}

    for epoch in range(1, epochs + 1):
        # Unfreeze encoder after warmup
        if epoch == freeze_epochs + 1:
            logger.info("Unfreezing encoder")
            model.unfreeze_encoder()
            optimizer = torch.optim.Adam(
                model.get_param_groups(encoder_lr=encoder_lr, head_lr=head_lr),
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10,
            )

        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            gt_four_point = batch["four_point"].to(device)
            target_gcps = batch["target_gcps"].to(device)
            ref_gcps = batch["ref_gcps"].to(device)

            optimizer.zero_grad()

            pred = model(inputs)

            # Primary loss: GCP reprojection
            loss = gcp_reprojection_loss(pred, target_gcps, ref_gcps, working_size)

            # Optional auxiliary loss: displacement L1
            if displacement_weight > 0:
                disp_loss = masked_l1_loss(pred, gt_four_point)
                loss = loss + displacement_weight * disp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_error_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                target_gcps = batch["target_gcps"].to(device)
                ref_gcps = batch["ref_gcps"].to(device)

                pred = model(inputs)
                loss = gcp_reprojection_loss(pred, target_gcps, ref_gcps, working_size)
                gcp_err = compute_gcp_error(pred, target_gcps, ref_gcps, working_size)

                val_loss_sum += loss.item()
                val_error_sum += gcp_err
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_error = val_error_sum / max(val_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_gcp_error"].append(avg_val_error)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[-1]["lr"]

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val GCP Error: {avg_val_error:.2f}px | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpoint on best validation GCP error
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_gcp_error": avg_val_error,
                "stage": "real",
                "config": cfg,
            }, ckpt_dir / "best_homography_real.pth")
            logger.info(
                f"  -> Saved best model (val_gcp_error={avg_val_error:.2f}px)"
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Save history
    with open(log_dir / "history_real.json", "w") as f:
        json.dump(history, f)

    logger.info(
        f"Real-supervised training complete. "
        f"Best val GCP error: {best_val_error:.2f}px"
    )
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", default="1", choices=["1", "2", "real"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint to warm-start from (stage=real only)")
    args = parser.parse_args()

    if args.stage == "1":
        train_stage1(args.config)
    elif args.stage == "2":
        train_stage2(args.config)
    else:
        train_real(args.config, from_checkpoint=args.checkpoint)
