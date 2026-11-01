"""R1 臂：把细节分支里重复的 3x3 卷积整块替换成 ScConv（替换式）。

## 替换了什么

细节分支的 stage2 / stage3 每个都长这样::

    [ConvBNReLU(C_in -> C_out, 3x3, stride=2),   # 负责下采样
     ConvBNReLU(C_out -> C_out, 3x3),            # <- 替换
     ConvBNReLU(C_out -> C_out, 3x3)]            # <- 替换

只替换后面这些 stride=1、通道数不变的重复卷积，共 **4 个**：

    detail.stage2[1] (64ch)   detail.stage2[2] (64ch)
    detail.stage3[1] (128ch)  detail.stage3[2] (128ch)

block 结构为 ``ScConv(C) -> ReLU``：ScConv 输入输出通道一致、空间尺寸不变，
可以直接顶替同通道的 3x3 卷积；保留 ReLU 是为了不改变该位置的
"卷积 + 非线性"语义。

## 为什么这么设计

* ScConv 的卖点是"降冗余、降算力且不掉点"，只有**替换**才能兑现——附加式
  实测无收益（见 results2中断/ablation_summary.md：S0/D1/P1/P2/G1 全部在噪声内）。
* D2 已经证明"去掉残差、让 ScConv 直接承担该位置的输出"是唯一有正收益的
  配置（+0.50%）；R1 是这个结论的彻底版本。
* 细节分支承载高分辨率边缘纹理，是夹杂/划痕这类细结构最依赖的通路；而这里
  的重复 3x3 卷积正是通道冗余最集中的地方。
* 换成 ScConv 后参数与 FLOPs 都是**净下降**，因此可以报"等/低算力下更优"。

## 与 D2 的关系

本臂**不含** BGA 之后的附加 ScConv（``SC_CONV_PLACEMENT = ()``），这样与 B0
对比时"替换式"是唯一变量。想要"R1 + D2 组合"请用 configScConvR1D2。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_repl_R1_detail"
EXPERIMENT_NAME = "R1-replace-detail"
EXPERIMENT_NOTE = "替换式 R1：detail 分支 stage2/stage3 的 4 个重复 3x3 卷积 → ScConv。"

USE_SC_CONV = True
# 关闭附加式，只保留替换式，保证变量唯一
SC_CONV_PLACEMENT = ()
SC_CONV_REPLACE = ("detail.repeat",)
SC_CONV_REPLACE_KEEP_RELU = True


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
