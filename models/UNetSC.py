"""U-Net + ScConv 消融模型。

消融臂：
  baseline          原始 U-Net，不使用 ScConv
  encoder_deep      仅 down3 / down4 使用 ScConv
  bottleneck        仅 down4 使用 ScConv
  decoder_deep      仅 up1 / up2 使用 ScConv
  encoder_bottleneck down3 / down4 使用 ScConv
  all               所有 DoubleConv 块使用 ScConv

ScConv 采用 SRU（空间重构）+ CRU（通道重构）。为了保持接口兼容，
ScConvBlock 输入/输出通道数与普通卷积相同，DoubleConv 中逐个替换 3x3 卷积。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScConv(nn.Module):
    """Spatial and Channel Reconstruction Convolution (SRU + CRU)。"""
    def __init__(self, channels: int, group_size: int = 2, squeeze_ratio: int = 2,
                 gate_threshold: float = 0.5, window_size: int = 7):
        super().__init__()
        if channels < 2:
            raise ValueError("ScConv 要求 channels >= 2")
        groups = min(group_size, channels)
        while channels % groups:
            groups -= 1
        self.gn = nn.GroupNorm(groups, channels)
        self.gate_threshold = gate_threshold
        self.window_size = window_size
        self.squeeze_ratio = max(1, squeeze_ratio)
        squeezed = max(1, channels // self.squeeze_ratio)
        self.squeeze = nn.Conv2d(channels, squeezed, 1, bias=False)
        self.group_conv = nn.Conv2d(squeezed, squeezed, 3, padding=1, groups=1, bias=False)
        self.project = nn.Conv2d(squeezed, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SRU: 基于归一化权重产生空间门控，抑制冗余响应并保留互补特征。
        normalized = self.gn(x)
        weight = self.gn.weight / (self.gn.weight.abs().sum() + 1e-6)
        gate = torch.sigmoid(normalized * weight.view(1, -1, 1, 1))
        informative = (gate >= self.gate_threshold).to(x.dtype) * x
        suppressed = (gate < self.gate_threshold).to(x.dtype) * x
        spatial = informative + suppressed

        # CRU: squeeze -> spatial transform -> channel restore，与输入残差融合。
        y = self.squeeze(spatial)
        y = self.group_conv(y)
        y = self.project(y)
        return F.relu(self.bn(y) + x, inplace=True)


class DoubleConv(nn.Module):
    """两次 3x3 卷积；指定位置时将卷积替换为 ScConv。"""
    def __init__(self, in_channels: int, out_channels: int, use_scconv: bool = False):
        super().__init__()
        layers = []
        current = in_channels
        for _ in range(2):
            if use_scconv:
                adapter = nn.Identity() if current == out_channels else nn.Conv2d(current, out_channels, 1, bias=False)
                conv = nn.Sequential(adapter, ScConv(out_channels))
            else:
                conv = nn.Conv2d(current, out_channels, 3, padding=1, bias=False)
            layers.extend([conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])
            current = out_channels
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_scconv: bool = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, use_scconv)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_scconv: bool = False):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, use_scconv)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetSC(nn.Module):
    """可按实验臂选择 ScConv 位置的四层 U-Net。"""
    VALID_ABLATIONS = {
        "baseline", "encoder_deep", "bottleneck", "decoder_deep",
        "encoder_bottleneck", "all",
    }

    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_classes: int = 4, ablation: str = "baseline"):
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"未知消融臂 {ablation!r}；可选：{sorted(self.VALID_ABLATIONS)}")
        self.ablation = ablation
        c1, c2, c3, c4, c5 = (base_channels * k for k in (1, 2, 4, 8, 16))
        use = lambda name: ablation == "all" or (ablation == "encoder_deep" and name in {"down3", "down4"}) or (ablation == "bottleneck" and name == "down4") or (ablation == "decoder_deep" and name in {"up1", "up2"}) or (ablation == "encoder_bottleneck" and name in {"down3", "down4"})
        self.inc = DoubleConv(in_channels, c1, use("inc"))
        self.down1 = Down(c1, c2, use("down1"))
        self.down2 = Down(c2, c3, use("down2"))
        self.down3 = Down(c3, c4, use("down3"))
        self.down4 = Down(c4, c5, use("down4"))
        self.up1 = Up(c5 + c4, c4, use("up1"))
        self.up2 = Up(c4 + c3, c3, use("up2"))
        self.up3 = Up(c3 + c2, c2, use("up3"))
        self.up4 = Up(c2 + c1, c1, use("up4"))
        self.outc = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_unet(in_channels: int = 3, base_channels: int = 32, num_classes: int = 4,
               ablation: str | None = None) -> UNetSC:
    import os
    selected = ablation or os.environ.get("UNETSC_ABLATION", "baseline")
    return UNetSC(in_channels, base_channels, num_classes, selected)



