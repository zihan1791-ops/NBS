"""C2b 臂：fused 位置放**普通 3x3 卷积块**，是 D2 的第二个归因对照。

C2a 控制的是"同容量"方向（1x1 块，参数只有 ScConv 的 55%）。C2b 控制的是
"同感受野"方向：``Conv3x3 + BN + ReLU``，参数量 147,712、FLOPs 184.32 M，
分别是 ScConv(128) 的 491% 与 497%。

这一臂的作用是给出归因的上界：
  * D2 > C2b → ScConv 在**更少参数、更少算力**下打赢普通 3x3 卷积，是最强的写法；
  * C2b > D2 → ScConv 在精度上并不占优，只在算力上占优，论文应转向轻量化叙事，
    并补测真实推理延迟（FLOPs 不等于延迟）。
"""
from pathlib import Path

from configScConvD2 import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_control_C2b_conv3x3"
EXPERIMENT_NAME = "C2b-conv3x3-control"
EXPERIMENT_NOTE = "D2 归因对照：fused 位置的 ScConv 换成 Conv3x3+BN+ReLU（无残差）。"

SC_CONV_FUSED_BLOCK = "conv3x3"


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
