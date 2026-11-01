from pathlib import Path
from configScConv import *
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_scconv_config2"
EXPERIMENT_NAME = "configScConv2"
EXPERIMENT_NOTE = "推荐后续组：降低初始学习率并保持 ScConv 结构不变。"
EPOCHS=350
BATCH_SIZE=32
LEARNING_RATE=5e-4
WEIGHT_DECAY=1e-4
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=4
EARLY_STOPPING_PATIENCE=30
SC_CONV_RESIDUAL_SCALE=0.1

def to_dict():
    return {name: str(value) if isinstance(value, Path) else value for name,value in globals().copy().items() if name.isupper()}

