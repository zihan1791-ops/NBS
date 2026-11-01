"""适配不同输入尺寸的 BiSeNetV2 语义分割网络。

网络包含细节分支、语义分支和双边引导聚合模块（BGA），主分割头与辅助分割头
都会将输出缩放到输入尺寸，以适配 NEU-Seg 图像。
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    AUXILIARY_HEAD_CHANNELS,
    DETAIL_CHANNELS,
    DETAIL_STAGE_REPEATS,
    GE_EXPANSION,
    HEAD_DROPOUT,
    INPUT_CHANNELS,
    MAIN_HEAD_CHANNELS,
    NUM_CLASSES,
    SEMANTIC_CHANNELS,
    STAGE_REPEATS,
    USE_AUXILIARY_HEADS,
)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class StemBlock(nn.Module):
    """BiSeNetV2 语义分支的输入模块，输出步长为 4。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        middle_channels = max(1, out_channels // 2)
        self.conv = ConvBNReLU(in_channels, out_channels, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(out_channels, middle_channels, 1, padding=0),
            ConvBNReLU(middle_channels, out_channels, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_channels * 2, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fuse(torch.cat((self.left(x), self.right(x)), dim=1))


class GELayerS1(nn.Module):
    """保持空间分辨率的聚合扩展层。"""

    def __init__(
        self, in_channels: int, out_channels: int, expansion: int = GE_EXPANSION
    ) -> None:
        super().__init__()
        mid_channels = in_channels * expansion
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3)
        self.dwconv = ConvBNReLU(
            in_channels, mid_channels, 3, groups=in_channels
        )
        self.project = ConvBNReLU(mid_channels, out_channels, 1, padding=0, activation=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ConvBNReLU(in_channels, out_channels, 1, padding=0, activation=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.project(self.dwconv(self.conv1(x)))
        return self.relu(x + residual)


class GELayerS2(nn.Module):
    """将特征图下采样一半的聚合扩展层。"""

    def __init__(
        self, in_channels: int, out_channels: int, expansion: int = GE_EXPANSION
    ) -> None:
        super().__init__()
        mid_channels = in_channels * expansion
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3)
        self.dwconv1 = ConvBNReLU(
            in_channels,
            mid_channels,
            3,
            stride=2,
            groups=in_channels,
            activation=False,
        )
        self.dwconv2 = ConvBNReLU(mid_channels, mid_channels, 3, groups=mid_channels)
        self.project = ConvBNReLU(mid_channels, out_channels, 1, padding=0, activation=False)
        self.shortcut = nn.Sequential(
            ConvBNReLU(
                in_channels,
                in_channels,
                3,
                stride=2,
                groups=in_channels,
                activation=False,
            ),
            ConvBNReLU(in_channels, out_channels, 1, padding=0, activation=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        x = self.project(x)
        return self.relu(x + residual)


class DetailBranch(nn.Module):

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        channels=DETAIL_CHANNELS,
        stage_repeats=DETAIL_STAGE_REPEATS,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or len(stage_repeats) != 3:
            raise ValueError("detail branch needs three stage widths and repeat counts")
        if any(repeat < 1 for repeat in stage_repeats):
            raise ValueError("each detail stage must contain at least one convolution")
        channel1, channel2, channel3 = channels
        self.stage1 = self._make_stage(in_channels, channel1, stage_repeats[0])
        self.stage2 = self._make_stage(channel1, channel2, stage_repeats[1])
        self.stage3 = self._make_stage(channel2, channel3, stage_repeats[2])

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, repeats: int):
        layers = [ConvBNReLU(in_channels, out_channels, 3, stride=2)]
        layers.extend(ConvBNReLU(out_channels, out_channels, 3) for _ in range(repeats - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage3(self.stage2(self.stage1(x)))


class SemanticBranch(nn.Module):
    """较窄且较深的语义分支，用于提取上下文语义特征。"""

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        channels=SEMANTIC_CHANNELS,
        expansion: int = GE_EXPANSION,
        stage_repeats=STAGE_REPEATS,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(stage_repeats) != 3:
            raise ValueError("semantic channels need four widths and stage repeats need three values")
        if any(repeat < 0 for repeat in stage_repeats):
            raise ValueError("stage repeats must be non-negative")
        self.channels = tuple(channels)
        self.stem = StemBlock(in_channels, channels[0])
        self.stage3 = self._make_stage(channels[0], channels[1], stage_repeats[0], expansion)
        self.stage4 = self._make_stage(channels[1], channels[2], stage_repeats[1], expansion)
        self.stage5 = self._make_stage(channels[2], channels[3], stage_repeats[2], expansion)
        self.context = ContextEmbeddingBlock(channels[3])

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, repeats: int, expansion: int):
        layers = [GELayerS2(in_channels, out_channels, expansion)]
        layers.extend(
            GELayerS1(out_channels, out_channels, expansion) for _ in range(repeats)
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        feat2 = self.stem(x)        # 输出步长为 4
        feat3 = self.stage3(feat2)  # 输出步长为 8
        feat4 = self.stage4(feat3)  # 输出步长为 16
        feat5 = self.stage5(feat4)  # 输出步长为 32
        semantic = self.context(feat5)
        return feat2, feat3, feat4, feat5, semantic


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.conv_gap = ConvBNReLU(channels, channels, 1, padding=0)
        self.conv_last = ConvBNReLU(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = F.adaptive_avg_pool2d(x, output_size=1)
        context = self.conv_gap(self.bn(context))
        return self.conv_last(x + context)


class BilateralGuidedAggregation(nn.Module):
    """通过双边空间门控融合细节特征和语义特征。"""

    def __init__(self, channels: int = DETAIL_CHANNELS[-1]) -> None:
        super().__init__()
        self.detail_main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.semantic_main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.fuse = ConvBNReLU(channels, channels, 3)

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        detail_size = detail.shape[-2:]
        detail_main = self.detail_main(detail)
        detail_gate = self.detail_gate(detail)
        semantic_gate = self.semantic_gate(semantic)
        semantic_main = self.semantic_main(semantic)

        semantic_gate = F.interpolate(
            semantic_gate, size=detail_size, mode="bilinear", align_corners=False
        )
        detail_fused = detail_main * torch.sigmoid(semantic_gate)

        semantic_fused = detail_gate * torch.sigmoid(semantic_main)
        semantic_fused = F.interpolate(
            semantic_fused, size=detail_size, mode="bilinear", align_corners=False
        )
        return self.fuse(detail_fused + semantic_fused)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_classes: int,
        dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels, 3)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, output_size) -> torch.Tensor:
        x = self.classifier(self.dropout(self.conv(x)))
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class BiSeNetV2(nn.Module):
    """带可选辅助训练头的 BiSeNetV2 分割模型。

    训练时返回主输出和辅助输出，用于深度监督；评估时只返回主输出。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        auxiliary: bool = USE_AUXILIARY_HEADS,
        main_head_channels: int = MAIN_HEAD_CHANNELS,
        in_channels: int = INPUT_CHANNELS,
        detail_channels=DETAIL_CHANNELS,
        detail_stage_repeats=DETAIL_STAGE_REPEATS,
        semantic_channels=SEMANTIC_CHANNELS,
        expansion: int = GE_EXPANSION,
        stage_repeats=STAGE_REPEATS,
        auxiliary_head_channels: int = AUXILIARY_HEAD_CHANNELS,
        dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        if detail_channels[-1] != semantic_channels[-1]:
            raise ValueError("detail and semantic output channels must match for BGA fusion")
        self.auxiliary = auxiliary
        self.detail = DetailBranch(in_channels, detail_channels, detail_stage_repeats)
        self.semantic = SemanticBranch(
            in_channels, semantic_channels, expansion, stage_repeats
        )
        self.bga = BilateralGuidedAggregation(detail_channels[-1])
        self.head = SegmentationHead(
            detail_channels[-1], main_head_channels, num_classes, dropout
        )

        if auxiliary:
            self.aux_heads = nn.ModuleList(
                [
                    SegmentationHead(width, auxiliary_head_channels, num_classes, dropout)
                    for width in semantic_channels
                ]
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        output_size = x.shape[-2:]
        detail = self.detail(x)
        feat2, feat3, feat4, feat5, semantic = self.semantic(x)
        fused = self.bga(detail, semantic)
        logits = self.head(fused, output_size)

        if self.training and self.auxiliary:
            aux_features = (feat2, feat3, feat4, feat5)
            aux_logits = [
                head(feature, output_size)
                for head, feature in zip(self.aux_heads, aux_features)
            ]
            return {"out": logits, "aux": aux_logits}
        return logits


def build_bisenetv2(
    num_classes: int = NUM_CLASSES,
    auxiliary: bool = USE_AUXILIARY_HEADS,
    main_head_channels: int = MAIN_HEAD_CHANNELS,
    in_channels: int = INPUT_CHANNELS,
    detail_channels=DETAIL_CHANNELS,
    detail_stage_repeats=DETAIL_STAGE_REPEATS,
    semantic_channels=SEMANTIC_CHANNELS,
    expansion: int = GE_EXPANSION,
    stage_repeats=STAGE_REPEATS,
    auxiliary_head_channels: int = AUXILIARY_HEAD_CHANNELS,
    dropout: float = HEAD_DROPOUT,
) -> BiSeNetV2:
    return BiSeNetV2(
        num_classes=num_classes,
        auxiliary=auxiliary,
        main_head_channels=main_head_channels,
        in_channels=in_channels,
        detail_channels=detail_channels,
        detail_stage_repeats=detail_stage_repeats,
        semantic_channels=semantic_channels,
        expansion=expansion,
        stage_repeats=stage_repeats,
        auxiliary_head_channels=auxiliary_head_channels,
        dropout=dropout,
    )
