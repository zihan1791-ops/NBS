"""Experiment-specific settings for BiSeNetV2 cross-branch decorrelation."""
from pathlib import Path

from config import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_branch_orthogonality"
EXPERIMENT_NAME = "BiSeNetV2-branch-orthogonality"
EXPERIMENT_NOTE = (
    "BiSeNetV2 baseline plus a warm-started cross-branch feature decorrelation loss; "
    "all standard segmentation and auxiliary losses are retained."
)
USE_BRANCH_ORTHOGONALITY = True
BRANCH_ORTHOGONALITY_WEIGHT = 0.01
BRANCH_ORTHOGONALITY_WARMUP_EPOCHS = 10


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
