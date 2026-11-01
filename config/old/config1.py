from pathlib import Path
from config import *
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_config1"
EXPERIMENT_NAME = "config1"
EXPERIMENT_NOTE = "中等学习率起步的 BiSeNetV2 对照组。"
EPOCHS=300
BATCH_SIZE=32
LEARNING_RATE=7e-4
WEIGHT_DECAY=1e-4
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=4
EARLY_STOPPING_PATIENCE=25
AUXILIARY_LOSS_WEIGHT=0.4
HEAD_DROPOUT=0.1

def to_dict():
    return {name: str(value) if isinstance(value, Path) else value for name,value in globals().copy().items() if name.isupper()}
