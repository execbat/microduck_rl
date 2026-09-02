"""Observation specifications for the Microduck velocity_swizzle task.

Same terms as ``tasks/velocity_rollers`` except ``head_command``, which
carries the REAL ``head_pose`` command on both groups (replaces the
zero-padded slot -- this task adds a real head-pose command, see
``microduck_commands_cfg.py``). ``body_command`` stays zero-padded
(no body-pose control here).
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.tasks.velocity import mdp

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

_HEAD_COMMAND = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})


@configclass
class MicroduckPolicyCfg(_RollersPolicyCfg):
    head_command: ObsTerm | None = _HEAD_COMMAND


@configclass
class MicroduckCriticCfg(_RollersCriticCfg):
    head_command: ObsTerm | None = _HEAD_COMMAND


@configclass
class MicroduckObservationsCfg(_RollersObservationsCfg):
    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
