"""D1 臂：把残差初始尺度 0.1 → 1.0。

现配置里 ScConv 被 x + 0.1 * ScConv(x) 包着，初始几乎不参与前向；而官方 SRU
门控满足 w1 + w2 = reweight <= 1，本身就不保能量。两者叠加后梯度几乎全走恒等
捷径。本臂把尺度初值提到 1.0，检验"模块是不是根本没学过东西"。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_D1_scale1"
EXPERIMENT_NAME = "D1-residual-scale-1.0"
EXPERIMENT_NOTE = "支线 D1：只改 SC_CONV_RESIDUAL_SCALE 0.1 → 1.0。"

SC_CONV_RESIDUAL = True
SC_CONV_RESIDUAL_SCALE = 1.0


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
