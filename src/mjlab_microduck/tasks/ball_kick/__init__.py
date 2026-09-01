"""Microduck BallKick task.

Public API (unchanged from the pre-refactor single-file module, so
``tasks/__init__.py`` needs no changes beyond the import path):

    from mjlab_microduck.tasks.ball_kick import make_microduck_ball_kick_env_cfg, MicroduckBallKickRlCfg
"""

from .microduck_ball_kick_env_cfg import MicroduckBallKickEnvCfg, make_microduck_ball_kick_env_cfg
from .microduck_rl_cfg import MicroduckBallKickRlCfg

__all__ = [
    "make_microduck_ball_kick_env_cfg",
    "MicroduckBallKickRlCfg",
    "MicroduckBallKickEnvCfg",
]
