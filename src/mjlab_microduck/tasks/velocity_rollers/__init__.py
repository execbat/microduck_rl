"""Microduck velocity_rollers (roller skate) task.

Public API (unchanged from the pre-refactor single-file module, so
``tasks/__init__.py`` needs no changes beyond the import path):

    from mjlab_microduck.tasks.velocity_rollers import make_microduck_velocity_rollers_env_cfg, MicroduckRollersRlCfg
"""

from .microduck_rl_cfg import MicroduckRollersRlCfg
from .microduck_velocity_rollers_env_cfg import (
    MicroduckVelocityRollersEnvCfg,
    make_microduck_velocity_rollers_env_cfg,
)

__all__ = [
    "make_microduck_velocity_rollers_env_cfg",
    "MicroduckRollersRlCfg",
    "MicroduckVelocityRollersEnvCfg",
]
