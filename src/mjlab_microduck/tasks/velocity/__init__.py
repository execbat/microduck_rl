"""Microduck velocity (walking) task.

Public API (unchanged from the pre-refactor single-file module, so
``tasks/__init__.py`` needs no changes beyond the import path):

    from mjlab_microduck.tasks.velocity import make_microduck_velocity_env_cfg, MicroduckRlCfg
"""

from .microduck_flags import BODY_POSE_CMD_RESAMPLE_S, HEAD_BODY_NAMES, HEAD_POSE_CMD_RESAMPLE_S
from .microduck_rl_cfg import MicroduckRlCfg
from .microduck_scene_cfg import MICRODUCK_ROUGH_TERRAINS_CFG
from .microduck_velocity_env_cfg import (
    MicroduckVelocityFlatEnvCfg,
    MicroduckVelocityFlatEnvCfg_PLAY,
    MicroduckVelocityRoughEnvCfg,
    MicroduckVelocityRoughEnvCfg_PLAY,
    make_microduck_velocity_env_cfg,
)

__all__ = [
    "make_microduck_velocity_env_cfg",
    "MicroduckRlCfg",
    "MicroduckVelocityFlatEnvCfg",
    "MicroduckVelocityFlatEnvCfg_PLAY",
    "MicroduckVelocityRoughEnvCfg",
    "MicroduckVelocityRoughEnvCfg_PLAY",
    # Re-exported for the other (not-yet-restructured) microduck_*_env_cfg.py
    # task modules, which used to import these from the old monolithic
    # tasks/microduck_velocity_env_cfg.py.
    "MICRODUCK_ROUGH_TERRAINS_CFG",
    "HEAD_BODY_NAMES",
    "HEAD_POSE_CMD_RESAMPLE_S",
    "BODY_POSE_CMD_RESAMPLE_S",
]
