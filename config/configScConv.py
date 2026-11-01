"""BiSeNetV2 + ScConv 配置，沿用统一 2904/726/840 数据划分。"""
from pathlib import Path
from config import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_scconv"
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 6
EARLY_STOPPING_PATIENCE = 30
USE_OFFICIAL_TEST_AS_VALIDATION = False
USE_SC_CONV = True
SC_CONV_GROUP_NUM = 4
SC_CONV_GATE_THRESHOLD = 0.5
SC_CONV_ALPHA = 0.5
SC_CONV_SQUEEZE_RATIO = 2
SC_CONV_GROUP_SIZE = 2
SC_CONV_GROUP_KERNEL_SIZE = 3
SC_CONV_PLACEMENT = ("fused",)
SC_CONV_RESIDUAL = True
SC_CONV_RESIDUAL_SCALE = 0.1
# fused 位置放什么：scconv（默认）/ conv1x1 / conv3x3 / relu（后三者是归因对照）
SC_CONV_FUSED_BLOCK = "scconv"


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
