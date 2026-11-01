"""Configuration for Exp-4: context + fusion."""
from .base import *  # noqa: F401,F403

EXPERIMENT_NAME = 'Exp-4: context + fusion'
USE_MULTI_SCALE_CONTEXT = True
USE_DEFECT_AWARE_BGA = True
USE_BOUNDARY_SUPERVISION = False


def to_dict():
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in globals().copy().items()
        if name.isupper()
    }
