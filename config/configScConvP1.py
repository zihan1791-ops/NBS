"""P1 臂：ScConv 只作用于细节分支输出（25x25x128）。

细节分支承载高分辨率边缘/纹理信息，是薄而细的夹杂与划痕最依赖的通路；
相比只挂在 BGA 之后，这里更接近"冗余所在的位置"。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_P1_detail"
EXPERIMENT_NAME = "P1-placement-detail"
EXPERIMENT_NOTE = "支线 P1：SC_CONV_PLACEMENT=('detail',)，其余同 S0。"

SC_CONV_PLACEMENT = ("detail",)


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
