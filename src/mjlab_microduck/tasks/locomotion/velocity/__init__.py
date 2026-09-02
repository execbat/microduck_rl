"""Robot-independent velocity locomotion configs for mjlab.

Choose a terrain-neutral, flat, or rough base, then supply robot assets,
sensor frames, and task-specific manager terms in a concrete task package.
"""

from .velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
    LocomotionVelocityFlatEnvCfg,
    LocomotionVelocityRoughEnvCfg,
)

__all__ = [
    "LocomotionVelocityEnvCfg",
    "LocomotionVelocityFlatEnvCfg",
    "LocomotionVelocityRoughEnvCfg",
]
