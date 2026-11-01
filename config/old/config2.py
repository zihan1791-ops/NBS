from pathlib import Path
from config import *
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_config2"
EXPERIMENT_NAME = "config2"
EXPERIMENT_NOTE = "较低学习率并延长收敛观察窗口。"
EPOCHS=350
BATCH_SIZE=32
LEARNING_RATE=5e-4
WEIGHT_DECAY=1e-4
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=5
EARLY_STOPPING_PATIENCE=30
AUXILIARY_LOSS_WEIGHT=0.35
HEAD_DROPOUT=0.1

def to_dict():
    return {name: str(value) if isinstance(value, Path) else value for name,value in globals().copy().items() if name.isupper()}
