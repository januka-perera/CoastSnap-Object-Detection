"""Training loop for heatmap regression model."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from .dataset import LandmarkDataset
from .model import HeatmapRegressor
from .heatmap_utils import soft_argmax_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    alpha: float = 20.0,
) -> torch.Tensor:
    """Weighted MSE loss that upweights pixels near landmarks.

    Args:
        pred: (B, N, H, W) predicted heatmaps.
        target: (B, N, H, W) ground-truth heatmaps.
        visibility: (B, N) boolean mask for visible landmarks.
        alpha: Upweighting factor for landmark regions.

    Returns:
        Scalar loss.
    """
    weight = 1.0 + alpha * target
    sq_diff = (pred - target) ** 2
    weighted_sq_diff = weight * sq_diff  # (B, N, H, W)

    # Mask out invisible landmarks
    vis_mask = visibility.float().unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    weighted_sq_diff = weighted_sq_diff * vis_mask

    num_visible = visibility.float().sum().clamp(min=1.0)
    H, W = pred.shape[2], pred.shape[3]
    loss = weighted_sq_diff.sum() / (num_visible * H * W)
    return loss


def js_divergence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Jensen-Shannon divergence treating heatmaps as probability distributions.

    Encourages the predicted heatmap to have the same shape/sharpness as the GT.

    Args:
        pred: (B, N, H, W) predicted heatmaps.
        target: (B, N, H, W) ground-truth heatmaps.
        visibility: (B, N) boolean mask for visible landmarks.

    Returns:
        Scalar loss.
    """
    B, N, H, W = pred.shape
    eps = 1e-8

    # Normalize to probability distributions per landmark
    pred_flat = pred.view(B, N, -1)
    target_flat = target.view(B, N, -1)

    pred_dist = pred_flat / (pred_flat.sum(dim=-1, keepdim=True) + eps)
    target_dist = target_flat / (target_flat.sum(dim=-1, keepdim=True) + eps)

    # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
    m = 0.5 * (pred_dist + target_dist)
    kl_pm = (pred_dist * (torch.log(pred_dist + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (target_dist * (torch.log(target_dist + eps) - torch.log(m + eps))).sum(dim=-1)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm  # (B, N)

    vis_mask = visibility.float()
    num_visible = vis_mask.sum().clamp(min=1.0)
    loss = (jsd * vis_mask).sum() / num_visible
    return loss


def coordinate_regression_loss(
    pred_heatmaps: torch.Tensor,
    gt_coords: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Coordinate regression loss via differentiable soft-argmax.

    Directly penalizes wrong peak locations by backpropagating through
    the soft-argmax coordinate extraction. This is the key loss that
    teaches the model WHERE to place heatmap peaks.

    Args:
        pred_heatmaps: (B, N, H, W) predicted heatmaps.
        gt_coords: (B, N, 2) normalised GT coordinates in [0, 1].
        visibility: (B, N) boolean mask for visible landmarks.

    Returns:
        Scalar loss (mean squared error on normalised coordinates).
    """
    pred_coords = soft_argmax_coordinates(pred_heatmaps)  # (B, N, 2)
    sq_diff = (pred_coords - gt_coords) ** 2  # (B, N, 2)
    sq_dist = sq_diff.sum(dim=-1)  # (B, N)

    vis_mask = visibility.float()
    num_visible = vis_mask.sum().clamp(min=1.0)
    loss = (sq_dist * vis_mask).sum() / num_visible
    return loss


def combined_loss(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    gt_coords: torch.Tensor,
    visibility: torch.Tensor,
    alpha: float = 5.0,
    coord_weight: float = 5.0,
) -> torch.Tensor:
    """Combined heatmap MSE + coordinate regression loss.

    Heatmap MSE teaches the model the right heatmap shape.
    Coordinate loss teaches the model WHERE to place the peaks.
    """
    heatmap_loss = weighted_mse_loss(pred_heatmaps, target_heatmaps, visibility, alpha)
    coord_loss = coordinate_regression_loss(pred_heatmaps, gt_coords, visibility)
    return heatmap_loss + coord_weight * coord_loss


def compute_euclidean_error(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    visibility: torch.Tensor,
    orig_widths: torch.Tensor,
    orig_heights: torch.Tensor,
) -> torch.Tensor:
    """Compute Euclidean pixel error at original resolution.

    Args:
        pred_coords: (B, N, 2) normalised predicted coordinates.
        gt_coords: (B, N, 2) normalised ground-truth coordinates.
        visibility: (B, N) boolean mask.
        orig_widths: (B,) original image widths.
        orig_heights: (B,) original image heights.

    Returns:
        Mean Euclidean error in pixels (scalar).
    """
    # Scale to original resolution
    scale = torch.stack([orig_widths.float(), orig_heights.float()], dim=-1)  # (B, 2)
    scale = scale.unsqueeze(1)  # (B, 1, 2)

    pred_px = pred_coords * scale
    gt_px = gt_coords * scale

    dist = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))  # (B, N)
    vis_mask = visibility.float()
    num_visible = vis_mask.sum().clamp(min=1.0)
    mean_error = (dist * vis_mask).sum() / num_visible
    return mean_error


def train(config_path: str = "configs/config.yaml"):
    """Main training function."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    full_dataset = LandmarkDataset(
        annotations_file=cfg["data"]["annotations_file"],
        images_dir=cfg["data"]["images_dir"],
        num_landmarks=cfg["data"]["num_landmarks"],
        input_size=(cfg["data"]["input_width"], cfg["data"]["input_height"]),
        output_size=(cfg["data"]["output_width"], cfg["data"]["output_height"]),
        sigma=cfg["heatmap"]["sigma"],
        augment=False,
    )

    # Split dataset
    n = len(full_dataset)
    seed = cfg["data_split"]["random_seed"]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    n_train = int(n * cfg["data_split"]["train_ratio"])
    n_val = int(n * cfg["data_split"]["val_ratio"])

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Save split for reproducibility
    split_info = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    with open(log_dir / "data_split.json", "w") as f:
        json.dump(split_info, f)
    logger.info(f"Split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    # Create augmented training dataset
    train_dataset = LandmarkDataset(
        annotations_file=cfg["data"]["annotations_file"],
        images_dir=cfg["data"]["images_dir"],
        num_landmarks=cfg["data"]["num_landmarks"],
        input_size=(cfg["data"]["input_width"], cfg["data"]["input_height"]),
        output_size=(cfg["data"]["output_width"], cfg["data"]["output_height"]),
        sigma=cfg["heatmap"]["sigma"],
        augment=cfg["augmentation"]["enabled"],
        aug_config=cfg["augmentation"],
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    model = HeatmapRegressor(
        num_landmarks=cfg["data"]["num_landmarks"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # Training config
    epochs = cfg["training"]["epochs"]
    freeze_epochs = cfg["training"]["freeze_encoder_epochs"]
    encoder_lr = cfg["training"]["encoder_lr"]
    decoder_lr = cfg["training"]["decoder_lr"]
    weight_decay = cfg["training"]["weight_decay"]
    alpha = cfg["loss"]["alpha"]
    coord_weight = cfg["loss"].get("coord_weight", 5.0)
    use_amp = cfg["training"]["use_mixed_precision"] and device.type == "cuda"

    # Start with frozen encoder
    model.freeze_encoder()
    optimizer = torch.optim.Adam(
        model.get_param_groups(encoder_lr=0.0, decoder_lr=decoder_lr),
        weight_decay=weight_decay,
    )

    if cfg["training"]["lr_scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg["training"]["plateau_factor"],
            patience=cfg["training"]["plateau_patience"],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = cfg["training"]["early_stopping_patience"]
    history = {"train_loss": [], "val_loss": [], "val_error_px": []}

    for epoch in range(1, epochs + 1):
        # Unfreeze encoder after warmup
        if epoch == freeze_epochs + 1:
            logger.info("Unfreezing encoder")
            model.unfreeze_encoder()
            optimizer = torch.optim.Adam(
                model.get_param_groups(encoder_lr=encoder_lr, decoder_lr=decoder_lr),
                weight_decay=weight_decay,
            )
            if cfg["training"]["lr_scheduler"] == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=cfg["training"]["plateau_factor"],
                    patience=cfg["training"]["plateau_patience"],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - freeze_epochs
                )

        # Training
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            heatmaps_gt = batch["heatmaps"].to(device)
            visibility = batch["visibility"].to(device)
            keypoints_gt = batch["keypoints"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                heatmaps_pred = model(images)
                loss = combined_loss(
                    heatmaps_pred, heatmaps_gt, keypoints_gt,
                    visibility, alpha, coord_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_error_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                heatmaps_gt = batch["heatmaps"].to(device)
                visibility = batch["visibility"].to(device)
                keypoints_gt = batch["keypoints"].to(device)
                orig_w = batch["orig_width"].to(device)
                orig_h = batch["orig_height"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    heatmaps_pred = model(images)
                    loss = combined_loss(
                        heatmaps_pred, heatmaps_gt, keypoints_gt,
                        visibility, alpha, coord_weight,
                    )

                pred_coords = soft_argmax_coordinates(heatmaps_pred)
                error = compute_euclidean_error(
                    pred_coords, keypoints_gt, visibility, orig_w, orig_h
                )

                val_loss_sum += loss.item()
                val_error_sum += error.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_error = val_error_sum / max(val_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_error_px"].append(avg_val_error)

        # LR scheduling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val Error: {avg_val_error:.1f}px | "
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
                "val_error_px": avg_val_error,
                "config": cfg,
            }, ckpt_dir / "best_model.pth")
            logger.info(f"  -> Saved best model (val_loss={avg_val_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Save training history
    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f)

    logger.info("Training complete.")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
