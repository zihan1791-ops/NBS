"""R1+D2 组合臂：替换式 R1 再加上已被验证的 D2（BGA 之后无残差 ScConv）。

## 为什么要这个臂

D2（``fused`` 位置去掉残差，ScConv 直接承担 BGA 输出）已经用 3 个种子确认
+0.50% test mIoU（见 results2中断/ablation_summary.md），是目前唯一成立的配置。
R1 是"替换式"的彻底版本。两者作用在不同位置、机制一致（都让 ScConv 直接承担
该位置的输出，不挂恒等旁路），因此**有可能是可叠加的**。

* 如果本臂 > R1 且 > D2 → 说明两个位置互补，论文可以报组合方案。
* 如果本臂 ≈ R1 → 说明收益已经饱和，BGA 位置是冗余的，报 R1 更简洁。
* 如果本臂 < R1 → 说明两处替换互相干扰，应只保留一处。

本臂是 R1/R2 之外的**可选**补充，优先级低于 R1，建议在前两者有结果后再决定。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_repl_R1_plus_D2"
EXPERIMENT_NAME = "R1D2-replace-detail+fused"
EXPERIMENT_NOTE = "组合臂：detail 替换式 + BGA 之后无残差 ScConv（D2）。"

USE_SC_CONV = True
SC_CONV_REPLACE = ("detail.repeat",)
SC_CONV_REPLACE_KEEP_RELU = True
# D2 的做法：fused 位置直接替换，不加残差旁路
SC_CONV_PLACEMENT = ("fused",)
SC_CONV_RESIDUAL = False


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
