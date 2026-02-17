"""Deep homography estimation network.

ResNet18 backbone with modified first conv for 2-channel grayscale input
(stacked reference + target), followed by global average pooling and a
fully-connected regression head predicting 8 corner displacements.

Input:  (B, 2, 384, 512) — stacked grayscale ref + target
Output: (B, 8) — 4-corner (dx, dy) displacements
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .model import BACKBONE_CHANNELS


class HomographyNet(nn.Module):
    """Deep homography estimation with ResNet18 backbone.

    Architecture:
        - ResNet18 encoder (modified conv1 for 2-channel input)
        - Global average pooling
        - FC(512 -> 256 -> 8) regression head
        - Pretrained init: average RGB conv weights into 2 channels
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 dropout: float = 0.5):
        super().__init__()

        ch = BACKBONE_CHANNELS[backbone]
        enc_out = ch[-1]  # 512 for resnet18

        # Load pretrained backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)

        # Modify first conv: 3-channel -> 2-channel
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            2, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            # Average RGB weights into 2 channels
            with torch.no_grad():
                # old_conv.weight shape: (64, 3, 7, 7)
                rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                new_conv.weight.copy_(rgb_mean.expand(-1, 2, -1, -1))

        # Build encoder
        self.encoder = nn.Sequential(
            new_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Global average pooling + regression head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 2, H, W) stacked grayscale reference and target.

        Returns:
            (B, 8) predicted 4-corner displacements.
        """
        features = self.encoder(x)
        pooled = self.pool(features)
        return self.head(pooled)

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_param_groups(self, encoder_lr: float, head_lr: float) -> list:
        """Return parameter groups with different learning rates."""
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]
