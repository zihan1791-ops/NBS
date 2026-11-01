from pathlib import Path
from configScConv import *
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_scconv_config1"
EXPERIMENT_NAME = "configScConv1"
EXPERIMENT_NOTE = "复现已有 ScConv 实验参数。"
EPOCHS=300
BATCH_SIZE=32
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=6
EARLY_STOPPING_PATIENCE=30
SC_CONV_RESIDUAL_SCALE=0.1

def to_dict():
    return {name: str(value) if isinstance(value, Path) else value for name,value in globals().copy().items() if name.isupper()}

