"""C2a 臂：fused 位置放**普通 1x1 卷积块**，是 D2 的归因对照。

D2（BGA 融合后直插 ScConv、无残差）是唯一有正收益的配置（+0.50%）。但"在融合
特征后多加一个块"本身就会带来额外容量与非线性，所以 D2 的增益有两种解释：

  (a) ScConv 的结构性归纳偏置（空间重构 + 通道重构）真的有用；
  (b) 任何同尺寸的卷积块接在融合特征后都同样有用。

C2a 把 ScConv 换成 ``Conv1x1 + BN + ReLU``，**其余逐项不变**：同位置、同无残差、
同 seed、同协议、同初始化方案。参数量 16,640、37.12 → 20.48 MFLOPs，约为
ScConv(128) 的 55%。

判定（见 REPORT 里的实验方案）：
  * D2 显著高于 C2a（差 > 合并 2σ）→ 归因成立，增益可归给 ScConv；
  * 两者在合并 1σ 内 → 归因不成立，增益应归给"融合后精化"这个**位置**。
"""
from pathlib import Path

from configScConvD2 import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_control_C2a_conv1x1"
EXPERIMENT_NAME = "C2a-conv1x1-control"
EXPERIMENT_NOTE = "D2 归因对照：fused 位置的 ScConv 换成 Conv1x1+BN+ReLU（无残差）。"

SC_CONV_FUSED_BLOCK = "conv1x1"


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
