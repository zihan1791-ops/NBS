"""P3 臂：两路各自先做 ScConv 细化，再交给原版 BGA 融合。

这一臂同时充当主线 X（跨流 ScConv 替代 BGA）的"公平对照"：BGA 结构保持原样，
只把 ScConv 前置到两条支流，从而保证与 X1 的差异只来自融合方式本身。
可作为低优先级臂按算力决定是否运行。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_P3_detail_semantic"
EXPERIMENT_NAME = "P3-placement-detail+semantic"
EXPERIMENT_NOTE = "支线 P3：SC_CONV_PLACEMENT=('detail','semantic')，BGA 保持原样。"

SC_CONV_PLACEMENT = ("detail", "semantic")


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
