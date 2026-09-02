"""Observation specifications for the Microduck roller_standup task.

Same observation TERMS as ``tasks/velocity_rollers`` -- required for
runtime obs-layout parity (a hard-swap between this policy and the roller
policy must not change the buffer shape). The only difference is
``nan_policy="sanitize"`` on both groups (same rationale/fix as
``roller_slope``: a rare contact divergence must not kill training via
rsl_rl's ``check_nan``).
"""

from mjlab_microduck.tasks.velocity_rollers.microduck_observations_cfg import (
    MicroduckCriticCfg as _RollersCriticCfg,
)
from mjlab_microduck.tasks.velocity_rollers.microduck_observations_cfg import (
    MicroduckObservationsCfg as _RollersObservationsCfg,
)
from mjlab_microduck.tasks.velocity_rollers.microduck_observations_cfg import (
    MicroduckPolicyCfg as _RollersPolicyCfg,
)
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckPolicyCfg(_RollersPolicyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.nan_policy = "sanitize"


@configclass
class MicroduckCriticCfg(_RollersCriticCfg):
    def __post_init__(self):
        super().__post_init__()
        self.nan_policy = "sanitize"


@configclass
class MicroduckObservationsCfg(_RollersObservationsCfg):
    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
