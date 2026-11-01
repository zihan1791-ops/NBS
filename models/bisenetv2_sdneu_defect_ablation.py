"""BiSeNetV2 ablation model for SD900 and NEU-Seg industrial defect segmentation."""

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
    USE_BOUNDARY_SUPERVISION,
    USE_DEFECT_AWARE_BGA,
    USE_MULTI_SCALE_CONTEXT,
)
from bisenetv2 import BiSeNetV2, ConvBNReLU


class LightweightMultiScaleContext(nn.Module):
    """Three low-cost receptive fields with a residual output."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = max(8, channels // 4)
        self.reduce = ConvBNReLU(channels, branch_channels, 1, padding=0)
        self.local = ConvBNReLU(branch_channels, branch_channels, 1, padding=0)
        self.dilated2 = self._depthwise_dilated(branch_channels, dilation=2)
        self.dilated4 = self._depthwise_dilated(branch_channels, dilation=4)
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _depthwise_dilated(channels: int, dilation: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        context = torch.cat(
            (self.local(reduced), self.dilated2(reduced), self.dilated4(reduced)),
            dim=1,
        )
        return x + self.residual_scale * self.project(context)


class DefectAwareBilateralFusion(nn.Module):
    """Spatially and channel-adaptive replacement for the original BGA."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 4)
        self.gate = nn.Sequential(
            ConvBNReLU(channels * 2, hidden, kernel_size=1, padding=0),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=True),
        )
        self.refine = ConvBNReLU(channels, channels, kernel_size=3)

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        semantic = F.interpolate(
            semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        batch, channels, height, width = detail.shape
        weights = self.gate(torch.cat((detail, semantic), dim=1))
        weights = weights.reshape(batch, 2, channels, height, width).softmax(dim=1)
        fused = weights[:, 0] * detail + weights[:, 1] * semantic
        return self.refine(fused)


class BiSeNetV2SDNEUDefectAblation(BiSeNetV2):
    """BiSeNetV2 with opt-in context, fusion, and boundary ablations."""

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
        use_multiscale_context: bool = USE_MULTI_SCALE_CONTEXT,
        use_defect_aware_bga: bool = USE_DEFECT_AWARE_BGA,
        use_boundary_supervision: bool = USE_BOUNDARY_SUPERVISION,
    ) -> None:
        super().__init__(
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
        self.use_multiscale_context = use_multiscale_context
        self.use_defect_aware_bga = use_defect_aware_bga
        self.use_boundary_supervision = use_boundary_supervision

        extra_modules = []
        if use_multiscale_context:
            self.multiscale_context = LightweightMultiScaleContext(semantic_channels[-1])
            extra_modules.append(self.multiscale_context)
        if use_defect_aware_bga:
            if detail_channels[-1] != semantic_channels[-1]:
                raise ValueError("detail and semantic channels must match for adaptive fusion")
            self.bga = DefectAwareBilateralFusion(detail_channels[-1])
            extra_modules.append(self.bga)
        if use_boundary_supervision:
            boundary_channels = max(32, detail_channels[1] // 2)
            self.boundary_head = nn.Sequential(
                ConvBNReLU(detail_channels[1], boundary_channels, kernel_size=3),
                nn.Conv2d(boundary_channels, 1, kernel_size=1, bias=True),
            )
            extra_modules.append(self.boundary_head)
        else:
            self.boundary_head = None

        self._initialize_extra_modules(extra_modules)

    @staticmethod
    def _initialize_extra_modules(modules) -> None:
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        output_size = x.shape[-2:]
        detail_stride2 = self.detail.stage1(x)
        detail_stride4 = self.detail.stage2(detail_stride2)
        detail_stride8 = self.detail.stage3(detail_stride4)

        feat2, feat3, feat4, feat5, semantic = self.semantic(x)
        if self.use_multiscale_context:
            semantic = self.multiscale_context(semantic)
        fused = self.bga(detail_stride8, semantic)
        logits = self.head(fused, output_size)

        result: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]
        if self.training and self.auxiliary:
            aux_features = (feat2, feat3, feat4, feat5)
            aux_logits = [
                head(feature, output_size)
                for head, feature in zip(self.aux_heads, aux_features)
            ]
            result = {"out": logits, "aux": aux_logits}
        else:
            result = logits

        if self.training and self.use_boundary_supervision:
            boundary_logits = self.boundary_head(detail_stride4)
            boundary_logits = F.interpolate(
                boundary_logits, size=output_size, mode="bilinear", align_corners=False
            )
            if not isinstance(result, dict):
                result = {"out": logits, "aux": []}
            result["boundary"] = boundary_logits
        return result


def build_bisenetv2_sdneu_defect_ablation(
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
    use_multiscale_context: bool = USE_MULTI_SCALE_CONTEXT,
    use_defect_aware_bga: bool = USE_DEFECT_AWARE_BGA,
    use_boundary_supervision: bool = USE_BOUNDARY_SUPERVISION,
) -> BiSeNetV2SDNEUDefectAblation:
    return BiSeNetV2SDNEUDefectAblation(
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
        use_multiscale_context=use_multiscale_context,
        use_defect_aware_bga=use_defect_aware_bga,
        use_boundary_supervision=use_boundary_supervision,
    )
