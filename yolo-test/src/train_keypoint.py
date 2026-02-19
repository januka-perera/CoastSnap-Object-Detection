"""
Two-phase training for the keypoint heatmap model.

Phase 1  Backbone frozen  →  decoder only   (lr=1e-3, 30 epochs)
Phase 2  Last ResNet block unfrozen          (lr=1e-4, 20 epochs)

Loss: weighted MSE — pixels where target > 0.1 receive higher weight to
      focus learning on the Gaussian peak region.

Usage
-----
    python train_keypoint.py --config ../configs/config.yaml
    python train_keypoint.py --config ../configs/config.yaml --resume ./keypoint_checkpoints/keypoint_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Allow running as a script from src/ or from the yolo-test root
sys.path.insert(0, str(Path(__file__).parent))

from dataset       import KeypointCropDataset
from model         import KeypointHeatmapModel
from heatmap_utils import hard_argmax


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """
    MSE loss with boosted weight on positive (peak) pixels.

    Parameters
    ----------
    pred, target : (B, 1, H, W)
    pos_weight   : Weight multiplier where target > 0.1.
    """
    mask    = (target > 0.1).float()
    weights = 1.0 + (pos_weight - 1.0) * mask
    return ((pred - target) ** 2 * weights).mean()


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: optim.Optimizer,
    device: torch.device,
    scaler,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        images   = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)

        optimiser.zero_grad()
        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            pred = model(images)
            loss = weighted_mse(pred, heatmaps)

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total += loss.item()

    return total / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Returns (val_loss, mean_EPE_in_heatmap_pixels).
    """
    model.eval()
    total_loss = 0.0
    total_epe  = 0.0
    hm_h = hm_w = None

    for batch in loader:
        images   = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)
        gt_kp    = batch["keypoint"].to(device)   # (B, 2) normalised

        pred = model(images)
        total_loss += weighted_mse(pred, heatmaps).item()

        if hm_h is None:
            hm_h, hm_w = pred.shape[-2], pred.shape[-1]

        pred_kp = hard_argmax(pred)   # (B, 2) normalised
        scale   = torch.tensor([hm_w, hm_h], device=device, dtype=torch.float32)
        epe     = ((pred_kp - gt_kp) * scale).norm(dim=1).mean()
        total_epe += epe.item()

    n = max(len(loader), 1)
    return total_loss / n, total_epe / n


# ---------------------------------------------------------------------------
# Single training phase
# ---------------------------------------------------------------------------

def run_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    phase_cfg: dict,
    checkpoint_path: Path,
    patience: int,
    phase_name: str,
) -> float:
    """
    Train for one phase.  Saves best checkpoint.  Returns best val_loss.
    """
    lr     = phase_cfg["lr"]
    epochs = phase_cfg["epochs"]

    optimiser = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=max(3, patience // 3), factor=0.5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_loss       = float("inf")
    epochs_no_improv = 0

    print(f"\n--- {phase_name}  |  LR={lr}  |  Epochs={epochs} ---")
    print(f"    Trainable params: {model.trainable_params():,}")

    for epoch in range(1, epochs + 1):
        tr_loss          = train_epoch(model, train_loader, optimiser, device, scaler)
        val_loss, val_epe = validate(model, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"  Epoch {epoch:3d}/{epochs}"
            f"  train={tr_loss:.4f}"
            f"  val={val_loss:.4f}"
            f"  EPE={val_epe:.2f}px"
        )

        if val_loss < best_loss:
            best_loss        = val_loss
            epochs_no_improv = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    → checkpoint saved  (val_loss={val_loss:.4f})")
        else:
            epochs_no_improv += 1
            if epochs_no_improv >= patience:
                print(f"    → early stopping at epoch {epoch}")
                break

    # Restore best weights before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return best_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train keypoint heatmap model")
    _default_cfg = str(Path(__file__).parent.parent / "configs" / "config.yaml")
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--resume", default=None,
                        help="Resume from an existing checkpoint.")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    kp_cfg   = cfg["keypoint"]
    data_cfg = cfg["data"]

    crops_dir       = Path(data_cfg["crops_dir"])
    checkpoint_dir  = Path(kp_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "keypoint_best.pt"

    input_size   = tuple(kp_cfg["input_size"])
    heatmap_size = tuple(kp_cfg["heatmap_size"])
    sigma        = kp_cfg["sigma"]
    patience     = kp_cfg["training"]["patience"]
    batch_size   = kp_cfg["training"]["batch_size"]
    num_workers  = kp_cfg["training"].get("workers", 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = KeypointCropDataset(
        crops_dir / "train", input_size, heatmap_size, sigma, augment=True
    )
    val_ds = KeypointCropDataset(
        crops_dir / "val", input_size, heatmap_size, sigma, augment=False
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = KeypointHeatmapModel(
        pretrained=kp_cfg.get("pretrained", True),
        decoder_channels=kp_cfg.get("decoder_channels", [128, 64, 32]),
        freeze_backbone=True,
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from: {args.resume}")

    # ── Phase 1 : backbone frozen ─────────────────────────────────────────
    run_phase(
        model, train_loader, val_loader, device,
        phase_cfg=kp_cfg["training"]["phase1"],
        checkpoint_path=checkpoint_path,
        patience=patience,
        phase_name="Phase 1 (backbone frozen)",
    )

    # ── Phase 2 : unfreeze last ResNet block ──────────────────────────────
    model.unfreeze_last_block()

    run_phase(
        model, train_loader, val_loader, device,
        phase_cfg=kp_cfg["training"]["phase2"],
        checkpoint_path=checkpoint_path,
        patience=patience,
        phase_name="Phase 2 (last ResNet block unfrozen)",
    )

    print(f"\nTraining complete.  Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
