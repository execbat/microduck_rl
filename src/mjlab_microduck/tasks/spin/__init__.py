"""Microduck spin task.

Public API (unchanged from the pre-refactor single-file module, so
``tasks/__init__.py`` needs no changes beyond the import path):

    from mjlab_microduck.tasks.spin import make_microduck_spin_env_cfg, MicroduckSpinRlCfg
"""

from .microduck_spin_env_cfg import MicroduckSpinEnvCfg, make_microduck_spin_env_cfg
from .microduck_rl_cfg import MicroduckSpinRlCfg

__all__ = [
    "make_microduck_spin_env_cfg",
    "MicroduckSpinRlCfg",
    "MicroduckSpinEnvCfg",
]
