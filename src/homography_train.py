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
from .homography_utils import four_point_to_homography, warp_image


def _four_point_to_homography_torch(four_point: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Differentiable 4-point to 3x3 homography via DLT (batched).

    Args:
        four_point: (B, 8) corner displacements.
        size: (width, height) of the image.

    Returns:
        (B, 3, 3) homography matrices.
    """
    B = four_point.shape[0]
    w, h = size
    device = four_point.device

    # Canonical corners
    src = torch.tensor([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype=four_point.dtype, device=device)  # (4, 2)

    dst = src.unsqueeze(0) + four_point.view(B, 4, 2)  # (B, 4, 2)
    src = src.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 2)

    # Build DLT system: for each point pair (x,y) -> (x',y')
    # Two equations per point, 8 equations total for 8 unknowns (h33=1)
    ones = torch.ones(B, 4, 1, dtype=four_point.dtype, device=device)
    zeros = torch.zeros(B, 4, 3, dtype=four_point.dtype, device=device)

    sx, sy = src[..., 0:1], src[..., 1:2]  # (B, 4, 1)
    dx, dy = dst[..., 0:1], dst[..., 1:2]

    src_h = torch.cat([sx, sy, ones], dim=-1)  # (B, 4, 3)

    row1 = torch.cat([src_h, zeros, -dx * sx, -dx * sy, -dx], dim=-1)  # (B, 4, 9)
    row2 = torch.cat([zeros, src_h, -dy * sx, -dy * sy, -dy], dim=-1)  # (B, 4, 9)

    A = torch.cat([row1, row2], dim=1)  # (B, 8, 9)

    # Solve via SVD
    _, _, Vt = torch.linalg.svd(A)
    H = Vt[:, -1, :].view(B, 3, 3)
    H = H / H[:, 2:3, 2:3]  # normalise so h33 = 1
    return H


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


def masked_photometric_loss(
    warped: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked L1 photometric loss for self-supervised fine-tuning.

    Args:
        warped: (B, 1, H, W) target warped to reference frame.
        reference: (B, 1, H, W) reference image.
        mask: (B, H, W) static region mask (1=static, 0=dynamic).

    Returns:
        Scalar loss.
    """
    diff = torch.abs(warped.squeeze(1) - reference.squeeze(1))  # (B, H, W)
    masked_diff = diff * mask
    num_pixels = mask.sum().clamp(min=1.0)
    return masked_diff.sum() / num_pixels


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

    use_amp = device.type == "cuda"
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
                loss = masked_photometric_loss(warped_tensor, ref_batch, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    if args.stage == 1:
        train_stage1(args.config)
    else:
        train_stage2(args.config)
