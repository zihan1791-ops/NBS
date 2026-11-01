"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)

依赖: torch >= 1.9.0

说明:
    标准 U-Net 编解码结构，使用双三次(bilinear)插值上采样并按 skip 特征尺寸对齐，
    因此输入尺寸无需是 16 的整数倍（例如 200x200）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. 基础模块
# ============================================================================

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """最大池化下采样 + 双卷积"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样 + 与 skip 特征拼接 + 双卷积"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        # 上采样到 skip 特征的空间尺寸，兼容非 2 的整数次幂的输入尺寸
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================================
# 2. U-Net 完整网络
# ============================================================================

class UNet(nn.Module):
    """标准 U-Net 分割网络，4 层下采样 / 4 层上采样。"""

    def __init__(self, in_channels=3, base_channels=64, num_classes=1):
        super(UNet, self).__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # ---- Encoder ----
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        # ---- Decoder ----
        self.up1 = Up(c5 + c4, c4)
        self.up2 = Up(c4 + c3, c3)
        self.up3 = Up(c3 + c2, c2)
        self.up4 = Up(c2 + c1, c1)

        # ---- 输出层 ----
        self.outc = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # ---- Encoder ----
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ---- Decoder ----
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # ---- 输出 ----
        seg_map = self.outc(x)

        return seg_map


# ============================================================================
# 3. 模型构建函数
# ============================================================================

def build_unet(in_channels: int = 3, base_channels: int = 32, num_classes: int = 1) -> UNet:
    return UNet(in_channels, base_channels, num_classes)


# ============================================================================
# 4. 单元测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("U-Net 模型测试")
    print("=" * 60)

    # 创建模型
    model = build_unet(in_channels=3, base_channels=32)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # 前向传播测试（256x256 与 200x200 均需支持）
    for hw in [(256, 256), (200, 200)]:
        x = torch.randn(2, 3, hw[0], hw[1])
        seg_map = model(x)
        print(f"\n输入形状: {x.shape}")
        print(f"输出分割图: {seg_map.shape}")

    print("\n测试通过! 模型可正常运行。")
