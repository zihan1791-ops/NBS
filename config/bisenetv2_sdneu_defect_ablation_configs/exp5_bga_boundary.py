"""Configuration for Exp-5: fusion + boundary."""
from .base import *  # noqa: F401,F403

EXPERIMENT_NAME = 'Exp-5: fusion + boundary'
USE_MULTI_SCALE_CONTEXT = False
USE_DEFECT_AWARE_BGA = True
USE_BOUNDARY_SUPERVISION = True


def to_dict():
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in globals().copy().items()
        if name.isupper()
    }
