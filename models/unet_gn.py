"""
U-Net with Group Normalisation (instead of Batch Normalisation).
================================================================

Identical architecture to models.unet.UNet (4-level encoder--decoder, no
attention gates) but every BatchNorm2d is replaced by GroupNorm. Motivation:
BatchNorm is unstable with the small batch sizes and the strongly
date-dependent snow distributions of this dataset, and its running statistics
collapse at evaluation time. GroupNorm is independent of the batch and fixes
this, matching the normalisation used in ResUNet++ for a fair comparison.

This is a standalone module so the original BatchNorm U-Net (and every previous
experiment that depends on it) is left untouched.
"""

import torch
import torch.nn as nn


def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a safe number of groups (divides num_channels)."""
    ng = min(num_groups, num_channels)
    while num_channels % ng != 0 and ng > 1:
        ng -= 1
    return nn.GroupNorm(ng, num_channels)


class DoubleConvGN(nn.Module):
    """Two 3x3 convolutions, each followed by GroupNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout_p: float = 0.0, num_groups: int = 8):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            _gn(out_channels, num_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            _gn(out_channels, num_groups),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetGN(nn.Module):
    """
    U-Net (GroupNorm) for snow-depth regression.

    Input : (B, in_channels, 256, 256)
    Output: (B, 1, 256, 256)  -> snow depth in metres (no output activation)
    """

    def __init__(self,
                 in_channels: int = 22,
                 out_channels: int = 1,
                 features: list = None,
                 dropout_p: float = 0.0,
                 num_groups: int = 8):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.dconv_down1 = DoubleConvGN(in_channels, features[0], num_groups=num_groups)
        self.dconv_down2 = DoubleConvGN(features[0], features[1], num_groups=num_groups)
        self.dconv_down3 = DoubleConvGN(features[1], features[2], num_groups=num_groups)
        self.dconv_down4 = DoubleConvGN(features[2], features[3], num_groups=num_groups)  # bottleneck

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = DoubleConvGN(features[3] + features[2], features[2],
                                      dropout_p=dropout_p, num_groups=num_groups)
        self.dconv_up2 = DoubleConvGN(features[2] + features[1], features[1],
                                      dropout_p=dropout_p, num_groups=num_groups)
        self.dconv_up1 = DoubleConvGN(features[1] + features[0], features[0],
                                      dropout_p=dropout_p, num_groups=num_groups)

        self.conv_last = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.dconv_down1(x)
        e2 = self.dconv_down2(self.pool(e1))
        e3 = self.dconv_down3(self.pool(e2))
        b  = self.dconv_down4(self.pool(e3))

        d3 = self.dconv_up3(torch.cat([self.upsample(b),  e3], dim=1))
        d2 = self.dconv_up2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dconv_up1(torch.cat([self.upsample(d2), e1], dim=1))

        return self.conv_last(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
