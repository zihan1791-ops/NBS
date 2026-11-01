"""Shared training protocol for the isolated SD900/NEU-Seg defect ablation."""
from pathlib import Path

from config import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_sdneu_defect_ablation"
EXPERIMENT_NAME = "BiSeNetV2-SDNEU-defect-ablation"
EXPERIMENT_NOTE = "Isolated seven-arm ablation for SD900 and NEU-Seg."

USE_SDNEU_DEFECT_ABLATION = True
USE_MULTI_SCALE_CONTEXT = False
USE_DEFECT_AWARE_BGA = False
USE_BOUNDARY_SUPERVISION = False
BOUNDARY_LOSS_WEIGHT = 0.15
BOUNDARY_POS_WEIGHT = 4.0
BOUNDARY_F1_TOLERANCE_PX = 1


def to_dict():
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in globals().copy().items()
        if name.isupper()
    }
