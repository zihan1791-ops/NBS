"""B0 基线臂：关闭 ScConv，其余与新协议完全一致。

刻意走 train_scconv.py 这条代码路径（而不是 train_bisenet.py），使基线与所有
ScConv 臂共享同一份训练脚本、同一份数据管线、同一套随机种子——两者只差
``USE_SC_CONV`` 一个开关。这样"ScConv 的净效应"才是干净的。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_B0_baseline"
EXPERIMENT_NAME = "B0-baseline"
EXPERIMENT_NOTE = "消融基线：USE_SC_CONV=False，等价于原版 BiSeNetV2。"

USE_SC_CONV = False


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
