"""D2 臂：去掉残差，直接把 ScConv 串进前向。

对比 D1 用来区分两种可能：(a) 只是初始尺度太小、学不动；(b) 可学习 scale 本身
让优化器倾向"关掉"模块。本臂把 scale 从优化目标里彻底移除。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_D2_direct"
EXPERIMENT_NAME = "D2-no-residual"
EXPERIMENT_NOTE = "支线 D2：SC_CONV_RESIDUAL=False，fused 位置直插 ScConv。"

SC_CONV_RESIDUAL = False


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
