"""Microduck velstand task.

Public API (unchanged from the pre-refactor single-file module, so
``tasks/__init__.py`` needs no changes beyond the import path):

    from mjlab_microduck.tasks.velstand import make_microduck_velstand_env_cfg, MicroduckVelStandRlCfg
"""

from .microduck_velstand_env_cfg import (
    MicroduckVelstandFlatEnvCfg,
    MicroduckVelstandFlatEnvCfg_PLAY,
    MicroduckVelstandRoughEnvCfg,
    MicroduckVelstandRoughEnvCfg_PLAY,
    make_microduck_velstand_env_cfg,
)
from .microduck_rl_cfg import MicroduckVelStandRlCfg

__all__ = [
    "make_microduck_velstand_env_cfg",
    "MicroduckVelStandRlCfg",
    "MicroduckVelstandFlatEnvCfg",
    "MicroduckVelstandFlatEnvCfg_PLAY",
    "MicroduckVelstandRoughEnvCfg",
    "MicroduckVelstandRoughEnvCfg_PLAY",
]
