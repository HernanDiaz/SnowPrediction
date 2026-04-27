"""
ResUNet++ para prediccion de profundidad de nieve.
===================================================

Arquitectura moderna que combina:
  - Encoder residual con GroupNorm (estable con batch pequeno)
  - Squeeze-and-Excitation (SE) por bloque (atencion por canal)
  - ASPP bottleneck (contexto multi-escala: r=1,6,12,18)
  - Attention gates en las skip connections del decoder
  - Decoder residual con GroupNorm

Motivacion para snow depth:
  - ResNet blocks: gradiente estable con 54 tiles de entrenamiento
  - GroupNorm: BatchNorm colapsa con bs=8 y pocas stats. GN no depende del batch.
  - SE: TPI domina (44% RF importance). SE aprende a ponderar TPI mas que otros canales.
  - ASPP: captura acumulacion de nieve a distintas escalas espaciales (ladera, valle, cumbre).
  - AttnGate: suprime skip connections irrelevantes (ya probado en Attention U-Net).

Referencia: Jha et al. 2020, "ResUNet++: An Advanced Architecture for Medical Image Segmentation"
            + mejoras propias (SE, ASPP, GroupNorm en lugar de BN).

Parametros aprox:
  features=[32,64,128,256]:  ~3.5M params
  features=[64,128,256,512]: ~14M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Bloques basicos
# ─────────────────────────────────────────────────────────────────────────────

class GroupNormAct(nn.Sequential):
    """GroupNorm + ReLU. Reemplaza BatchNorm para mayor estabilidad con bs pequeno."""
    def __init__(self, channels: int, num_groups: int = 8):
        # Ajuste automatico si channels < num_groups
        ng = min(num_groups, channels)
        while channels % ng != 0 and ng > 1:
            ng -= 1
        super().__init__(
            nn.GroupNorm(ng, channels),
            nn.ReLU(inplace=True),
        )


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al. 2018).
    Aprende a reponderar cada canal de caracteristicas.
    Especialmente util cuando las features de entrada tienen importancias muy distintas
    (TPI >> DEM >> Slope >> Northness ~ Eastness en este dataset).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """
    Bloque residual con GroupNorm + SE.
    Conv3x3 -> GN -> ReLU -> Conv3x3 -> GN -> SE -> + skip
    """
    def __init__(self, in_ch: int, out_ch: int,
                 num_groups: int = 8, dropout_p: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            GroupNormAct(out_ch, num_groups),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_ch), out_ch),
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(num_groups, out_ch), out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.conv(x)) + self.skip(x)
        return self.drop(self.relu(out))


# ─────────────────────────────────────────────────────────────────────────────
# ASPP Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al. 2018, DeepLabv3).
    Extrae contexto a multiples escalas con dilataciones r=1,6,12,18.
    Para tiles 256x256 con patrones topograficos a distintas escalas espaciales.
    """
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        rates = [1, 6, 12, 18]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3,
                          padding=r, dilation=r, bias=False),
                GroupNormAct(out_ch, num_groups),
            )
            for r in rates
        ])
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            GroupNormAct(out_ch, num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        gap   = F.interpolate(self.gap(x), size=x.shape[2:],
                              mode='bilinear', align_corners=True)
        feats.append(gap)
        return self.fuse(torch.cat(feats, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Attention Gate (igual que Attention U-Net, Oktay 2018)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Conv2d(g_ch,    inter_ch, 1, bias=True)
        self.W_x = nn.Conv2d(x_ch,    inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1,        1, bias=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(g, size=x.shape[2:],
                             mode='bilinear', align_corners=True)
        psi  = torch.sigmoid(self.psi(F.relu(self.W_g(g_up) + self.W_x(x), inplace=True)))
        return x * psi


# ─────────────────────────────────────────────────────────────────────────────
# ResUNet++
# ─────────────────────────────────────────────────────────────────────────────

class ResUNetPP(nn.Module):
    """
    ResUNet++ para prediccion de snow depth.

    Entrada : (B, in_channels, 256, 256)
    Salida  : (B, 1, 256, 256)

    Args:
        in_channels:  Canales de entrada (5 sin SCE, 6 con SCE)
        out_channels: Canales de salida (1 = snow depth en metros)
        features:     Filtros por nivel [f1, f2, f3, f4] (encoder)
        dropout_p:    Dropout en bloques del decoder (0 = desactivado)
        num_groups:   Grupos para GroupNorm (default 8; se ajusta si channels < 8)
    """

    def __init__(self,
                 in_channels:  int = 5,
                 out_channels: int = 1,
                 features:     list = None,
                 dropout_p:    float = 0.0,
                 num_groups:   int = 8):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        f1, f2, f3, f4 = features

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ResidualBlock(in_channels, f1, num_groups)
        self.enc2 = ResidualBlock(f1,          f2, num_groups)
        self.enc3 = ResidualBlock(f2,          f3, num_groups)

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ASPP ───────────────────────────────────────────────────
        self.bottleneck_in  = ResidualBlock(f3, f4, num_groups)
        self.aspp           = ASPPModule(f4, f4, num_groups)
        self.bottleneck_out = ResidualBlock(f4, f4, num_groups)

        # ── Attention gates ───────────────────────────────────────────────────
        self.att3 = AttentionGate(f4, f3, f3 // 2)
        self.att2 = AttentionGate(f3, f2, f2 // 2)
        self.att1 = AttentionGate(f2, f1, f1 // 2)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec3 = ResidualBlock(f4 + f3, f3, num_groups, dropout_p)
        self.dec2 = ResidualBlock(f3 + f2, f2, num_groups, dropout_p)
        self.dec1 = ResidualBlock(f2 + f1, f1, num_groups, dropout_p)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ── Cabeza de salida ──────────────────────────────────────────────────
        self.head = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck con ASPP
        b = self.bottleneck_in(self.pool(e3))
        b = self.aspp(b)
        b = self.bottleneck_out(b)

        # Decoder con attention gates
        a3 = self.att3(b,  e3)
        d3 = self.dec3(torch.cat([self.upsample(b),  a3], dim=1))

        a2 = self.att2(d3, e2)
        d2 = self.dec2(torch.cat([self.upsample(d3), a2], dim=1))

        a1 = self.att1(d2, e1)
        d1 = self.dec1(torch.cat([self.upsample(d2), a1], dim=1))

        return self.head(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Test rapido
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = ResUNetPP(in_channels=5, out_channels=1, features=[64, 128, 256, 512])
    x = torch.randn(2, 5, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {model.count_parameters():,}")

    model_small = ResUNetPP(in_channels=5, out_channels=1, features=[32, 64, 128, 256])
    print(f"Params (small): {model_small.count_parameters():,}")
