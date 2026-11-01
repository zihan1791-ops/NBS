"""Configuration for Exp-6: full model."""
from .base import *  # noqa: F401,F403

EXPERIMENT_NAME = 'Exp-6: full model'
USE_MULTI_SCALE_CONTEXT = True
USE_DEFECT_AWARE_BGA = True
USE_BOUNDARY_SUPERVISION = True


def to_dict():
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in globals().copy().items()
        if name.isupper()
    }
