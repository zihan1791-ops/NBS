from pathlib import Path
from config import *
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_config3"
EXPERIMENT_NAME = "config3"
EXPERIMENT_NOTE = "适度增强正则化的探索组。"
EPOCHS=350
BATCH_SIZE=24
LEARNING_RATE=5e-4
WEIGHT_DECAY=3e-4
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=4
EARLY_STOPPING_PATIENCE=30
AUXILIARY_LOSS_WEIGHT=0.3
HEAD_DROPOUT=0.2

def to_dict():
    return {name: str(value) if isinstance(value, Path) else value for name,value in globals().copy().items() if name.isupper()}
