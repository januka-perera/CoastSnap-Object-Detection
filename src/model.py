"""Encoder-decoder model for heatmap regression using ResNet50 backbone."""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HeatmapRegressor(nn.Module):
    """ResNet50 encoder with U-Net-style decoder for heatmap regression.

    Input:  (B, 3, 384, 512)
    Output: (B, num_landmarks, 96, 128)
    """

    def __init__(self, num_landmarks: int = 5, pretrained: bool = True):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Encoder stages (extract feature maps at different scales)
        # Input 512x384 -> after each stage:
        # layer0 (conv1+bn+relu+pool): 128x96, 64ch
        # layer1: 128x96, 256ch
        # layer2: 64x48, 512ch
        # layer3: 32x24, 1024ch
        # layer4: 16x12, 2048ch
        self.encoder0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.encoder1 = resnet.layer1  # 128x96, 256ch
        self.encoder2 = resnet.layer2  # 64x48, 512ch
        self.encoder3 = resnet.layer3  # 32x24, 1024ch
        self.encoder4 = resnet.layer4  # 16x12, 2048ch

        # Decoder with skip connections
        # Up1: 16x12 -> 32x24, cat with encoder3 (1024ch)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(2048 + 1024, 256)
        self.drop1 = nn.Dropout2d(0.15)

        # Up2: 32x24 -> 64x48, cat with encoder2 (512ch)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(256 + 512, 128)
        self.drop2 = nn.Dropout2d(0.15)

        # Up3: 64x48 -> 128x96, cat with encoder1 (256ch)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(128 + 256, 64)

        # Final 1x1 conv to produce heatmaps
        self.head = nn.Conv2d(64, num_landmarks, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.encoder0(x)   # (B, 64, 96, 128) - after maxpool
        e1 = self.encoder1(e0)  # (B, 256, 96, 128)
        e2 = self.encoder2(e1)  # (B, 512, 48, 64)
        e3 = self.encoder3(e2)  # (B, 1024, 24, 32)
        e4 = self.encoder4(e3)  # (B, 2048, 12, 16)

        # Decoder
        d1 = self.up1(e4)                    # (B, 2048, 24, 32)
        d1 = torch.cat([d1, e3], dim=1)      # (B, 3072, 24, 32)
        d1 = self.drop1(self.dec1(d1))        # (B, 256, 24, 32)

        d2 = self.up2(d1)                     # (B, 256, 48, 64)
        d2 = torch.cat([d2, e2], dim=1)       # (B, 768, 48, 64)
        d2 = self.drop2(self.dec2(d2))        # (B, 128, 48, 64)

        d3 = self.up3(d2)                     # (B, 128, 96, 128)
        d3 = torch.cat([d3, e1], dim=1)       # (B, 384, 96, 128)
        d3 = self.dec3(d3)                     # (B, 64, 96, 128)

        out = self.head(d3)                   # (B, N, 96, 128)
        out = torch.sigmoid(out)
        return out

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for module in [self.encoder0, self.encoder1, self.encoder2,
                       self.encoder3, self.encoder4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for module in [self.encoder0, self.encoder1, self.encoder2,
                       self.encoder3, self.encoder4]:
            for param in module.parameters():
                param.requires_grad = True

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> list:
        """Return parameter groups with different learning rates."""
        encoder_params = []
        for module in [self.encoder0, self.encoder1, self.encoder2,
                       self.encoder3, self.encoder4]:
            encoder_params.extend(module.parameters())

        decoder_params = []
        for module in [self.up1, self.dec1, self.up2, self.dec2,
                       self.up3, self.dec3, self.head]:
            decoder_params.extend(module.parameters())

        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]
