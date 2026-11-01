"""P2 臂：ScConv 只作用于语义分支上下文输出（7x7x128）。

7x7 的空间规模配 128 个通道，是全网通道冗余最严重的表示；CRU 的通道重排
在这里理论上最有发挥空间。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_P2_semantic"
EXPERIMENT_NAME = "P2-placement-semantic"
EXPERIMENT_NOTE = "支线 P2：SC_CONV_PLACEMENT=('semantic',)，其余同 S0。"

SC_CONV_PLACEMENT = ("semantic",)


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
