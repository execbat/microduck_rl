"""Microduck roller_crouch task — crouch-glide trick on rollers.

One-shot trick triggered by button A via the runtime's --ground-pick slot:
the robot crouches and glides on its momentum (a ~1s hold), then rises and
hands control back to the roller policy. See ``microduck_flags.py`` for the
full design rationale and phase timing.

Hybrid task: rides on the same robot/sensors/obs layout as
``tasks/velocity_rollers`` but layers the ``ground_pick`` task's one-shot
phase-command machinery on top instead of continuous velocity tracking.
Flat terrain only, no play-specific behavior (``play`` is accepted and
ignored, matching the original file's dead parameter -- same situation as
``velocity_rollers``).
"""

from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.tasks.velocity.locomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from mjlab_microduck.tasks.velocity_rollers.microduck_observations_cfg import (
    MicroduckObservationsCfg as _RollersObservationsCfg,
)
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import FEET_GROUND_CFG, MICRODUCK_WALK_ROLLERS_ROBOT_CFG, SELF_COLLISION_CFG
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckRollerCrouchEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Microduck roller_crouch task -- flat terrain, crouch-glide trick."""

    # Observations are identical to velocity_rollers's (same robot, same
    # unified 61D layout, same IMU/delay/DR treatment) -- reused directly
    # rather than duplicated.
    observations: _RollersObservationsCfg = _RollersObservationsCfg()
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


def make_microduck_roller_crouch_env_cfg(play: bool = False):
    """Create the Microduck roller_crouch environment configuration.

    ``play`` is accepted (and unused) purely for signature parity with the
    old function of the same name and with ``velocity_rollers`` -- there's
    no play-specific behavior in this task.

    Kept as a drop-in replacement: same signature, same return type (a real
    ``mjlab.envs.ManagerBasedRlEnvCfg``, via ``.to_mjlab_cfg()``), so gym
    registration keeps working unmodified.
    """
    return MicroduckRollerCrouchEnvCfg().to_mjlab_cfg()
