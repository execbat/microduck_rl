"""Observation specifications for the Microduck spin task.

Identical to ``velocity_rollers``'s observations, except the critic's
privileged ``wheel_vel`` term matches ALL ``passive_*`` joints
(``r"^passive_.*"``) rather than just the wheel joints
(``r"^passive_.*wheel"``) -- a genuine, deliberate difference from the
original file, not a typo, so it's not reused wholesale like
``roller_crouch``'s identical observations were.
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks.velocity_rollers.microduck_observations_cfg import (
    MicroduckCriticCfg as _RollersCriticCfg,
)
from mjlab_microduck.tasks.velocity_rollers.microduck_observations_cfg import (
    MicroduckObservationsCfg as _RollersObservationsCfg,
)
from mjlab_microduck.tasks.velocity_rollers.microduck_observations_cfg import (
    MicroduckPolicyCfg,
)
from mjlab_microduck.utils.configclass import configclass

_ALL_PASSIVE = SceneEntityCfg("robot", joint_names=(r"^passive_.*",))


@configclass
class MicroduckCriticCfg(_RollersCriticCfg):
    wheel_vel: ObsTerm | None = ObsTerm(func=mdp.joint_vel_rel, scale=1.0, params={"asset_cfg": _ALL_PASSIVE})


@configclass
class MicroduckObservationsCfg(_RollersObservationsCfg):
    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
