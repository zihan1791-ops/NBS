"""Configuration for Baseline."""
from .base import *  # noqa: F401,F403

EXPERIMENT_NAME = 'Baseline'
USE_MULTI_SCALE_CONTEXT = False
USE_DEFECT_AWARE_BGA = False
USE_BOUNDARY_SUPERVISION = False


def to_dict():
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in globals().copy().items()
        if name.isupper()
    }
