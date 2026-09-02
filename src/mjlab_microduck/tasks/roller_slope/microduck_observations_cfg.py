"""Observation specifications for the Microduck roller_slope task.

Same observation TERMS as ``tasks/velocity_rollers`` (DR/obs/reset untouched
here) -- the only difference is ``nan_policy="sanitize"`` on both groups.

A rare contact event (~1 per 25M env-steps) can diverge the free joint into
NaN. Because of a one-substep offset, the ``nan_state`` termination only
catches it on the FOLLOWING step (reset), but the NaN has already reached
the CURRENT step's observation -- rsl_rl's ``check_nan`` then kills the run.
``nan_policy="sanitize"`` replaces NaN/Inf with 0 in the returned obs (no
crash); ``nan_state`` still resets the offending env right after.
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
