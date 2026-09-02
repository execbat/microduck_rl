"""Microduck velocity_rollers task — roller skate variant.

Migrated to mjlab 1.3.0 + canonical BAM (2026-07), matching the velocity
env's sim2real machinery, for the roller model: 14 actuated joints + 4
passive wheels (``passive_{L,R}{F,R}wheel``, two per blade, interspersed in
the joint order after each ankle — everything resolves joints by NAME, never
index). Legs run the canonical BAM actuator like every other variant.

Task design (the roller recipe):
  ``cmd_x`` semantics: 0 = coast, >0 = push to accelerate, <0 = brake.
  ``cmd[2]`` = heading error via ``RelativeHeadingVelocityCommand``.
  Sole positive task reward is ``wheel_speed`` — the robot must actually spin
  its wheels; braking/skating_air_time/forward_lean/heading_hold shape the
  skating style. See ``microduck_velocity_swizzle_env_cfg.py``'s sibling
  task (``tasks/velocity_swizzle/`` once migrated) for the classic-swizzle
  variant built on top of this one.

Obs migrated to the unified 61D layout (twist + zero-padded head/body
command slots) so roller policies load through the runtime's --new-cmd-obs
path. Symmetry OFF (SYMMETRY_CFG is hardcoded for the old 51D layout).
"""

from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.tasks.locomotion.velocity import LocomotionVelocityFlatEnvCfg
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import FEET_GROUND_CFG, MICRODUCK_WALK_ROLLERS_ROBOT_CFG, SELF_COLLISION_CFG
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckVelocityRollersEnvCfg(LocomotionVelocityFlatEnvCfg):
    """Microduck velocity_rollers task -- flat terrain, roller skate gait."""

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
        # NOTE: an env-side action clip was tried here to bound the target,
        # but the deployment pipeline (infer_policy.py) does NOT clip -> the
        # clip would only exist in sim, a train/deploy mismatch. The
        # over-command deterrent lives policy-side instead
        # (action_over_limit reward), baked into the network so it
        # transfers with the ONNX.


def make_microduck_velocity_rollers_env_cfg(play: bool = False):
    """Create the Microduck velocity_rollers environment configuration.

    ``play`` is accepted (and unused) purely for signature parity with the
    old function of the same name, which registration calls as
    ``make_microduck_velocity_rollers_env_cfg(play=True)`` for the play
    variant -- there's no play-specific behavior in this task.

    Kept as a drop-in replacement: same signature, same return type (a real
    ``mjlab.envs.ManagerBasedRlEnvCfg``, via ``.to_mjlab_cfg()``), so gym
    registration -- and the ``tasks/velocity_swizzle`` /
    ``tasks/roller_slope`` / ``tasks/roller_standup`` tasks built on top of
    it -- keep working unmodified.
    """
    return MicroduckVelocityRollersEnvCfg().to_mjlab_cfg()
