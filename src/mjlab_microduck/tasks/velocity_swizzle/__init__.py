"""Microduck velocity_swizzle task.

Public API (unchanged from the pre-refactor single-file module, so
``tasks/__init__.py`` needs no changes beyond the import path):

    from mjlab_microduck.tasks.velocity_swizzle import make_microduck_velocity_swizzle_env_cfg, MicroduckSwizzleRlCfg
"""

from .microduck_rl_cfg import MicroduckSwizzleRlCfg
from .microduck_velocity_swizzle_env_cfg import (
    MicroduckVelocitySwizzleEnvCfg,
    make_microduck_velocity_swizzle_env_cfg,
)

__all__ = [
    "make_microduck_velocity_swizzle_env_cfg",
    "MicroduckSwizzleRlCfg",
    "MicroduckVelocitySwizzleEnvCfg",
]
