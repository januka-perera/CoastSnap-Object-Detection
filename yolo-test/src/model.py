"""
Keypoint heatmap model: pretrained ResNet18 encoder + lightweight decoder.

Architecture
------------
Encoder  ResNet18 (pretrained ImageNet)
  enc0 : conv1 + bn + relu + maxpool  →  stride 4   (64 ch)
  enc1 : layer1                        →  stride 4   (64 ch)
  enc2 : layer2                        →  stride 8   (128 ch)
  enc3 : layer3                        →  stride 16  (256 ch)
  enc4 : layer4                        →  stride 32  (512 ch)

Decoder  progressive ×2 upsampling with skip connections
  dec4 : upsample + concat(enc3) → d1 channels
  dec3 : upsample + concat(enc2) → d2 channels
  dec2 : upsample + concat(enc1) → d3 channels
  head : upsample + conv → 1 channel + sigmoid

Output: (B, 1, H/4, W/4) ≈ (B, 1, 56, 56) for a 224×224 input.

Freeze control
--------------
  freeze_backbone=True          : fix entire encoder
  unfreeze_last_block()         : unfreeze enc4 only (for phase 2)
  unfreeze_all()                : unfreeze entire encoder (for phase 3 if desired)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample ×2 → concatenate skip → double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class KeypointHeatmapModel(nn.Module):
    """
    ResNet18 encoder + decoder for single-keypoint heatmap prediction.

    Parameters
    ----------
    pretrained       : Load ImageNet weights for the encoder.
    decoder_channels : Output channel counts for the three decoder blocks
                       [d1, d2, d3].  Defaults to [128, 64, 32].
    freeze_backbone  : Freeze all encoder parameters on construction.
    unfreeze_last_block : (only used when freeze_backbone=True) also keep
                          enc4 trainable.
    """

    def __init__(
        self,
        pretrained: bool = True,
        decoder_channels=None,
        freeze_backbone: bool = False,
        unfreeze_last_block: bool = False,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [128, 64, 32]

        # ── Encoder ──────────────────────────────────────────────────────
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        self.enc0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )                               # 64  ch, stride 4
        self.enc1 = backbone.layer1     # 64  ch, stride 4
        self.enc2 = backbone.layer2     # 128 ch, stride 8
        self.enc3 = backbone.layer3     # 256 ch, stride 16
        self.enc4 = backbone.layer4     # 512 ch, stride 32

        # ── Decoder ──────────────────────────────────────────────────────
        d1, d2, d3 = decoder_channels
        self.dec4 = DecoderBlock(512, 256, d1)   # skip ← enc3
        self.dec3 = DecoderBlock(d1,  128, d2)   # skip ← enc2
        self.dec2 = DecoderBlock(d2,   64, d3)   # skip ← enc1

        # Final head: one more ×2 upsample → (H/4, W/4) = (56, 56) for 224-px input
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(d3, d3),
            nn.Conv2d(d3, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._set_backbone_grad(not freeze_backbone)
        if freeze_backbone and unfreeze_last_block:
            self._set_layer_grad(self.enc4, True)

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _set_backbone_grad(self, requires_grad: bool):
        for part in (self.enc0, self.enc1, self.enc2, self.enc3, self.enc4):
            self._set_layer_grad(part, requires_grad)

    @staticmethod
    def _set_layer_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def freeze_backbone(self):
        """Freeze entire encoder."""
        self._set_backbone_grad(False)

    def unfreeze_last_block(self):
        """Freeze encoder except enc4 (last ResNet block)."""
        self._set_backbone_grad(False)
        self._set_layer_grad(self.enc4, True)

    def unfreeze_all(self):
        """Unfreeze entire encoder."""
        self._set_backbone_grad(True)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        heatmap : (B, 1, H/4, W/4)  values in [0, 1]
        """
        e0 = self.enc0(x)     # stride 4
        e1 = self.enc1(e0)    # stride 4
        e2 = self.enc2(e1)    # stride 8
        e3 = self.enc3(e2)    # stride 16
        e4 = self.enc4(e3)    # stride 32

        d = self.dec4(e4, e3)
        d = self.dec3(d,  e2)
        d = self.dec2(d,  e1)
        return self.head(d)
