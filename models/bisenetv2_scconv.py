"""在 BiSeNetV2 双边分支融合前插入 ScConv 的模型变体。

ScConv 按 SRU（空间重构）后接 CRU（通道重构）的顺序处理特征。
本变体分别细化细节分支输出和语义上下文输出，再交给原有 BGA 融合。
"""

from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bisenetv2 import BiSeNetV2, ConvBNReLU, GELayerS1
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


class SpatialReconstructionUnit(nn.Module):
    """使用 GroupNorm 权重构造门控，并交叉重组空间信息。"""

    def __init__(
        self,
        channels: int,
        group_num: int = 4,
        gate_threshold: float = 0.5,
        epsilon: float = 1e-6,
        gamma_normalise: bool = True,
    ) -> None:
        super().__init__()
        if channels % group_num != 0:
            raise ValueError(
                f"ScConv 通道数 {channels} 必须能被 GroupNorm 分组数 {group_num} 整除"
            )
        if channels % 2 != 0:
            raise ValueError(f"SRU 要求通道数为偶数，当前为 {channels}")
        if not 0.0 < gate_threshold < 1.0:
            raise ValueError("ScConv gate_threshold 必须位于 (0, 1) 内")

        self.group_norm = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.gate_threshold = gate_threshold
        self.epsilon = epsilon
        self.gamma_normalise = gamma_normalise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.group_norm(x)
        if self.gamma_normalise:
            # 官方实现的写法：γ / Σγ。注意 γ̂ ≈ 1/C，而 GroupNorm 输出是零均值单位
            # 方差，于是 sigmoid(γ̂·x) 会塌到 0.5 附近（C=128 时实测 std 仅 0.002、
            # 99.7% 的元素落在阈值 ±0.01 内），门控退化为固定的通道切分。
            gamma_sum = self.group_norm.weight.sum()
            gamma = self.group_norm.weight / (gamma_sum + self.epsilon)
        else:
            # 关闭归一化，让门控工作在可用的动态范围上（γ 本身就在 1 附近）。
            gamma = self.group_norm.weight
        reweight = torch.sigmoid(normalized * gamma.view(1, -1, 1, 1))

        informative = torch.where(
            reweight > self.gate_threshold,
            torch.ones_like(reweight),
            reweight,
        )
        less_informative = torch.where(
            reweight > self.gate_threshold,
            torch.zeros_like(reweight),
            reweight,
        )

        x1 = informative * x
        x2 = less_informative * x
        x11, x12 = torch.chunk(x1, chunks=2, dim=1)
        x21, x22 = torch.chunk(x2, chunks=2, dim=1)
        return torch.cat((x11 + x22, x12 + x21), dim=1)


class ChannelReconstructionUnit(nn.Module):
    """通过分流、变换和融合降低通道冗余。"""

    def __init__(
        self,
        channels: int,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("ScConv alpha 必须位于 (0, 1) 内")
        if squeeze_ratio < 1:
            raise ValueError("ScConv squeeze_ratio 必须大于等于 1")
        if group_kernel_size % 2 != 1:
            raise ValueError("ScConv group_kernel_size 请使用奇数以保持空间尺寸")

        upper_channels = int(alpha * channels)
        lower_channels = channels - upper_channels
        upper_squeezed = upper_channels // squeeze_ratio
        lower_squeezed = lower_channels // squeeze_ratio
        if min(upper_squeezed, lower_squeezed) < 1:
            raise ValueError("ScConv 通道压缩后至少要保留 1 个通道")
        if upper_squeezed % group_size != 0 or channels % group_size != 0:
            raise ValueError(
                "ScConv group_size 必须同时整除上分支压缩通道数和输出通道数"
            )

        self.upper_channels = upper_channels
        self.lower_channels = lower_channels
        self.upper_squeezed = upper_squeezed
        self.lower_squeezed = lower_squeezed

        self.squeeze_upper = nn.Conv2d(
            upper_channels, upper_squeezed, kernel_size=1, bias=False
        )
        self.squeeze_lower = nn.Conv2d(
            lower_channels, lower_squeezed, kernel_size=1, bias=False
        )

        self.groupwise_conv = nn.Conv2d(
            upper_squeezed,
            channels,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
            bias=True,
        )
        self.pointwise_upper = nn.Conv2d(
            upper_squeezed, channels, kernel_size=1, bias=False
        )
        self.pointwise_lower = nn.Conv2d(
            lower_squeezed,
            channels - lower_squeezed,
            kernel_size=1,
            bias=False,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upper, lower = torch.split(
            x, (self.upper_channels, self.lower_channels), dim=1
        )
        upper = self.squeeze_upper(upper)
        lower = self.squeeze_lower(lower)

        upper_features = self.groupwise_conv(upper) + self.pointwise_upper(upper)
        lower_features = torch.cat(
            (self.pointwise_lower(lower), lower), dim=1
        )
        fused = torch.cat((upper_features, lower_features), dim=1)

        channel_weights = F.softmax(self.global_pool(fused), dim=1)
        fused = channel_weights * fused
        first, second = torch.chunk(fused, chunks=2, dim=1)
        return first + second


class ScConv(nn.Module):
    """ScConv 模块：SRU 空间重构后接 CRU 通道重构。"""

    def __init__(
        self,
        channels: int,
        group_num: int = 4,
        gate_threshold: float = 0.5,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
        gamma_normalise: bool = True,
    ) -> None:
        super().__init__()
        self.sru = SpatialReconstructionUnit(
            channels,
            group_num=group_num,
            gate_threshold=gate_threshold,
            gamma_normalise=gamma_normalise,
        )
        self.cru = ChannelReconstructionUnit(
            channels,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cru(self.sru(x))


class ResidualScConv(nn.Module):
    """以可学习残差比例使用子模块，初始时尽量保留原始特征。

    ``module`` 并不限定为 ScConv：C2a/C2b 归因对照臂也用它把普通卷积块包成
    完全相同的残差形式，这样"块类型"是两组之间唯一的变量。
    """

    def __init__(self, module: nn.Module, initial_scale: float = 0.1) -> None:
        super().__init__()
        if initial_scale < 0.0:
            raise ValueError("ScConv 残差初始比例不能小于 0")
        self.module = module
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.module(x)


class BiSeNetV2ScConv(BiSeNetV2):
    """在 BiSeNetV2 上应用 ScConv，支持两种范式。

    1. **附加式**（``scconv_placement``）：对两路输出或 BGA 融合特征再叠加一个
       ScConv 分支。实测（见 results2中断/ablation_summary.md）这种用法没有收益：
       带残差时它近似恒等，即使把模块完全激活（D1）或把门控动态范围打开
       （G1）精度也不动。
    2. **替换式**（``scconv_replace``）：把网络里已有的标准 3x3 卷积整块换成
       ScConv，通道数不变，因此参数和 FLOPs 都下降。这与 ScConv 原论文的用法
       一致，实测去掉残差的替换式（D2）是唯一有正收益的配置。

    另外 ``scconv_fused_block`` 允许把 fused 位置上那个附加块换成**普通块**
    （1x1 卷积 / 3x3 卷积 / 纯 ReLU）。它们不包含任何 ScConv，专门用来回答
    "D2 那 +0.5% 到底是 ScConv 带来的，还是'在融合特征上多加一个块'带来的"。
    """

    REPLACE_TARGETS = ("detail.repeat", "semantic.ge_conv1")
    # fused 位置可选块类型；除 scconv 外都是归因对照用的普通块。
    FUSED_BLOCKS = ("scconv", "conv1x1", "conv3x3", "relu")

    def __init__(
        self,
        *args,
        scconv_enabled: bool = True,
        scconv_group_num: int = 4,
        scconv_gate_threshold: float = 0.5,
        scconv_alpha: float = 0.5,
        scconv_squeeze_ratio: int = 2,
        scconv_group_size: int = 2,
        scconv_group_kernel_size: int = 3,
        scconv_gamma_normalise: bool = True,
        scconv_placement: Sequence[str] = ("fused",),
        scconv_residual: bool = True,
        scconv_residual_scale: float = 0.1,
        scconv_replace: Sequence[str] = (),
        scconv_replace_keep_relu: bool = True,
        scconv_fused_block: str = "scconv",
        **kwargs,
    ) -> None:
        detail_channels = kwargs.get("detail_channels", DETAIL_CHANNELS)
        semantic_channels = kwargs.get("semantic_channels", SEMANTIC_CHANNELS)
        super().__init__(*args, **kwargs)
        placement = {item.lower() for item in scconv_placement}
        unknown = placement - {"detail", "semantic", "fused"}
        if unknown:
            raise ValueError(f"未知 ScConv 插入位置：{sorted(unknown)}")

        replace = {item.lower() for item in scconv_replace}
        unknown_replace = replace - set(self.REPLACE_TARGETS)
        if unknown_replace:
            raise ValueError(
                f"未知 ScConv 替换目标：{sorted(unknown_replace)}；"
                f"可选：{list(self.REPLACE_TARGETS)}"
            )

        fused_block = str(scconv_fused_block).lower()
        if fused_block not in self.FUSED_BLOCKS:
            raise ValueError(
                f"未知的 fused 块类型：{scconv_fused_block!r}；"
                f"可选：{list(self.FUSED_BLOCKS)}"
            )
        if fused_block != "scconv" and placement - {"fused"}:
            raise ValueError(
                "SC_CONV_FUSED_BLOCK 只作用于 fused 位置；对照臂请把 "
                "SC_CONV_PLACEMENT 设为 ('fused',)；若想同时保留 detail/semantic "
                "的 ScConv，请另开一臂而不是复用这一开关"
            )

        self.scconv_enabled = scconv_enabled
        self.scconv_placement = tuple(sorted(placement))
        self.scconv_residual = scconv_residual
        self.scconv_fused_block = fused_block
        self.scconv_replace = tuple(sorted(replace))
        self.scconv_replace_keep_relu = scconv_replace_keep_relu
        self.scconv_replaced_layers: List[str] = []
        channels = detail_channels[-1]
        if channels != semantic_channels[-1]:
            raise ValueError("BGA 输入通道需要一致，ScConv 变体要求两路输出通道相同")

        scconv_args = {
            "group_num": scconv_group_num,
            "gate_threshold": scconv_gate_threshold,
            "alpha": scconv_alpha,
            "squeeze_ratio": scconv_squeeze_ratio,
            "group_size": scconv_group_size,
            "group_kernel_size": scconv_group_kernel_size,
            "gamma_normalise": scconv_gamma_normalise,
        }
        self._scconv_args = scconv_args

        def make_scconv():
            module = ScConv(channels, **scconv_args)
            if scconv_residual:
                return ResidualScConv(module, initial_scale=scconv_residual_scale)
            return module

        def make_fused_block():
            """fused 位置的块；对照臂用普通块顶替 ScConv，其余一切不变。"""
            if fused_block == "scconv":
                return make_scconv()
            if fused_block == "relu":
                # 零参数对照：只多一层激活，用来排除"多一层非线性"这个解释。
                # 刻意不套残差（否则就引入了可学习参数，不再"零参数"）。
                return nn.ReLU(inplace=True)
            kernel_size = 1 if fused_block == "conv1x1" else 3
            module = ConvBNReLU(channels, channels, kernel_size=kernel_size)
            if scconv_residual:
                # 与 ScConv 臂保持一致：同样包成可学习残差比例。
                return ResidualScConv(module, initial_scale=scconv_residual_scale)
            return module

        self.detail_scconv = (
            make_scconv()
            if scconv_enabled and "detail" in placement
            else nn.Identity()
        )
        self.semantic_scconv = (
            make_scconv()
            if scconv_enabled and "semantic" in placement
            else nn.Identity()
        )
        self.fused_scconv = (
            make_fused_block()
            if scconv_enabled and "fused" in placement
            else nn.Identity()
        )

        if scconv_enabled and replace:
            self._apply_replacements(replace, scconv_replace_keep_relu)

    # ---- 替换式应用的实现 ----

    def _replacement_block(self, channels: int, keep_relu: bool) -> nn.Sequential:
        """构造替换 ConvBNReLU(C, C, 3) 的模块：ScConv 后接（可选的）ReLU。

        ScConv 输出通道数与输入一致、空间尺寸不变，因此可以直接顶替同通道的
        标准 3x3 卷积；保留 ReLU 是为了不改变该位置的"卷积 + 非线性"语义。
        """
        layers: List[nn.Module] = [ScConv(channels, **self._scconv_args)]
        if keep_relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _apply_replacements(self, targets, keep_relu: bool) -> None:
        if "detail.repeat" in targets:
            # 细节分支每个 stage 的首个卷积带 stride=2 负责下采样，只替换后续
            # 那些 stride=1、通道数不变的重复 3x3 卷积。
            for stage_name in ("stage2", "stage3"):
                stage = getattr(self.detail, stage_name)
                for index in range(1, len(stage)):
                    layer = stage[index]
                    if isinstance(layer, ConvBNReLU) and layer.conv.stride == (1, 1):
                        out_channels = layer.conv.out_channels
                        stage[index] = self._replacement_block(out_channels, keep_relu)
                        self.scconv_replaced_layers.append(
                            f"detail.{stage_name}[{index}] ({out_channels}ch)"
                        )
        if "semantic.ge_conv1" in targets:
            # 语义分支 GELayerS1 的 conv1 是通道数不变的 3x3 卷积，替换它影响
            # stride 8/16/32 三个尺度上的全部聚合扩展层。
            for stage_name in ("stage3", "stage4", "stage5"):
                stage = getattr(self.semantic, stage_name)
                for index, module in enumerate(stage):
                    if isinstance(module, GELayerS1):
                        out_channels = module.conv1.conv.out_channels
                        module.conv1 = self._replacement_block(out_channels, keep_relu)
                        self.scconv_replaced_layers.append(
                            f"semantic.{stage_name}[{index}].conv1 ({out_channels}ch)"
                        )
        if not self.scconv_replaced_layers:
            raise RuntimeError(f"替换目标 {sorted(targets)} 没有匹配到任何卷积层")
        # 新插入的卷积需要用同一套初始化，保证与未替换的层一致。
        self._init_weights()

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        output_size = x.shape[-2:]
        detail = self.detail_scconv(self.detail(x))
        feat2, feat3, feat4, feat5, semantic = self.semantic(x)
        semantic = self.semantic_scconv(semantic)
        fused = self.fused_scconv(self.bga(detail, semantic))
        logits = self.head(fused, output_size)

        if self.training and self.auxiliary:
            auxiliary_features = (feat2, feat3, feat4, feat5)
            auxiliary_logits = [
                head(feature, output_size)
                for head, feature in zip(self.aux_heads, auxiliary_features)
            ]
            return {"out": logits, "aux": auxiliary_logits}
        return logits


def build_bisenetv2_scconv(
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
    scconv_enabled: bool = True,
    scconv_group_num: int = 4,
    scconv_gate_threshold: float = 0.5,
    scconv_alpha: float = 0.5,
    scconv_squeeze_ratio: int = 2,
    scconv_group_size: int = 2,
    scconv_group_kernel_size: int = 3,
    scconv_gamma_normalise: bool = True,
    scconv_placement: Sequence[str] = ("fused",),
    scconv_residual: bool = True,
    scconv_residual_scale: float = 0.1,
    scconv_replace: Sequence[str] = (),
    scconv_replace_keep_relu: bool = True,
    scconv_fused_block: str = "scconv",
) -> BiSeNetV2ScConv:
    """按配置构造 BiSeNetV2 + ScConv。"""
    return BiSeNetV2ScConv(
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
        scconv_enabled=scconv_enabled,
        scconv_group_num=scconv_group_num,
        scconv_gate_threshold=scconv_gate_threshold,
        scconv_alpha=scconv_alpha,
        scconv_squeeze_ratio=scconv_squeeze_ratio,
        scconv_group_size=scconv_group_size,
        scconv_group_kernel_size=scconv_group_kernel_size,
        scconv_gamma_normalise=scconv_gamma_normalise,
        scconv_placement=scconv_placement,
        scconv_residual=scconv_residual,
        scconv_residual_scale=scconv_residual_scale,
        scconv_replace=scconv_replace,
        scconv_replace_keep_relu=scconv_replace_keep_relu,
        scconv_fused_block=scconv_fused_block,
    )
