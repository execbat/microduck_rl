"""Microduck spin task — fast rotation in place, on rollers.

Cyclic trick triggered by button A via the runtime's --ground-pick slot:
~1 turn counter-clockwise at ~3 rad/s then a clean stop, standing.

Hybrid task: rides on the same robot/sensors as ``tasks/velocity_rollers``
but layers ``roller_crouch``'s one-shot phase-command machinery on top,
driving a TARGET YAW RATE instead of a joint pose (see
``microduck_flags.py`` for the full design rationale).

Flat terrain only, no play-specific behavior (``play`` is accepted and
ignored, matching the original file's dead parameter -- same situation as
``velocity_rollers``/``roller_crouch``).
"""

from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.tasks.velocity.locomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import FEET_GROUND_CFG, MICRODUCK_WALK_ROLLERS_ROBOT_CFG, SELF_COLLISION_CFG
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckSpinEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Microduck spin task -- flat terrain, in-place rotation trick."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    def __post_init__(self):
        self.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
        self.scene.sensors = (FEET_GROUND_CFG, SELF_COLLISION_CFG)
        self.scene.terrain = TerrainEntityCfg(terrain_type="plane")
        self.viewer.body_name = "trunk_base"

        self.actions.joint_pos.scale = 1.0


def make_microduck_spin_env_cfg(play: bool = False):
    """Create the Microduck spin environment configuration.

    ``play`` is accepted (and unused) purely for signature parity with the
    old function of the same name -- there's no play-specific behavior in
    this task.

    Kept as a drop-in replacement: same signature, same return type (a real
    ``mjlab.envs.ManagerBasedRlEnvCfg``, via ``.to_mjlab_cfg()``), so gym
    registration keeps working unmodified.
    """
    return MicroduckSpinEnvCfg().to_mjlab_cfg()
