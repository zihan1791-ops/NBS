"""SD900 U-Net settings, kept separate from the NEU and shared configs."""
from pathlib import Path

from configUnetP300 import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_DIR / "data" / "SD-saliency"
DATASET_PATHS = dict(DATASET_PATHS)
DATASET_PATHS["sd-saliency"] = str(DATA_ROOT)
DATASET_DEFAULTS = {key: dict(value) for key, value in DATASET_DEFAULTS.items()}
DATASET_DEFAULTS["sd-saliency"] = dict(DATASET_DEFAULTS["sd"])
DATASET_DEFAULTS["sd-saliency"]["batch_size"] = 8
SAVE_DIR = str(PROJECT_DIR / "results2中断" / "SD900" / "unet")
