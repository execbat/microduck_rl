"""Microduck roller_slope task — passive balanced descent.

Robot spawns on flat ground (small forward impulse), rolls down a
descending ramp, and glides while staying balanced. No steering: the twist
command is neutralised (``rel_standing_envs=1.0``). Custom flat+ramp
terrain (``FlatRampTerrainCfg``), stiffness curriculum
(``terrain_levels_slope``). Unified 61D obs -> hot-swappable at runtime
(--new-cmd-obs) -- inherited as-is from ``tasks.velocity_rollers`` (DR/obs/
reset untouched here).

Rides on the same robot/sensors/base observation set as
``tasks/velocity_rollers`` -- subclasses its env cfg class directly
(the original file called ``make_microduck_velocity_rollers_env_cfg(play=play)``).
"""

from mjlab_microduck.tasks.velocity_rollers.microduck_velocity_rollers_env_cfg import MicroduckVelocityRollersEnvCfg
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import build_terrain
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckRollerSlopeEnvCfg(MicroduckVelocityRollersEnvCfg):
    """Microduck roller_slope task -- custom flat+ramp terrain."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    # Runtime-selected: which terrain (train mix vs a play-mode fixed/random
    # slope) gets built in __post_init__ -- see make_microduck_roller_slope_env_cfg().
    play: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = build_terrain(self.play)


def make_microduck_roller_slope_env_cfg(play: bool = False):
    """Create the Microduck roller_slope environment configuration.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration keeps working unmodified.
    """
    return MicroduckRollerSlopeEnvCfg(play=play).to_mjlab_cfg()
