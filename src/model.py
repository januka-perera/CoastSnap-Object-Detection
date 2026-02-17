"""Encoder-decoder model for heatmap regression using ResNet backbone."""

import torch
import torch.nn as nn
import torchvision.models as models

# Channel counts for each encoder stage by backbone
BACKBONE_CHANNELS = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
}


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
    """ResNet encoder with U-Net-style decoder for heatmap regression.

    Input:  (B, 3, 384, 512)
    Output: (B, num_landmarks, 96, 128)
    """

    def __init__(self, num_landmarks: int = 7, pretrained: bool = True,
                 backbone: str = "resnet50"):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Load backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)

        ch = BACKBONE_CHANNELS[backbone]
        # ch = (e0_out, e1_out, e2_out, e3_out, e4_out)

        self.encoder0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(ch[4] + ch[3], 256)
        self.drop1 = nn.Dropout2d(0.15)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(256 + ch[2], 128)
        self.drop2 = nn.Dropout2d(0.15)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(128 + ch[1], 64)

        # Final 1x1 conv to produce heatmaps
        self.head = nn.Conv2d(64, num_landmarks, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.drop1(self.dec1(d1))

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.drop2(self.dec2(d2))

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        out = self.head(d3)
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
