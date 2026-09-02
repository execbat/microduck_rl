"""Microduck roller_slope task.

Public API (unchanged from the pre-refactor single-file module, so
``tasks/__init__.py`` needs no changes beyond the import path):

    from mjlab_microduck.tasks.roller_slope import make_microduck_roller_slope_env_cfg, MicroduckRollerSlopeRlCfg
"""

from .microduck_roller_slope_env_cfg import MicroduckRollerSlopeEnvCfg, make_microduck_roller_slope_env_cfg
from .microduck_rl_cfg import MicroduckRollerSlopeRlCfg

__all__ = [
    "make_microduck_roller_slope_env_cfg",
    "MicroduckRollerSlopeRlCfg",
    "MicroduckRollerSlopeEnvCfg",
]
