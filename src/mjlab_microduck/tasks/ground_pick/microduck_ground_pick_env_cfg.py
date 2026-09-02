"""Microduck ground_pick task — reach the mouth to the ground, then stand back up.

Episodic policy that crouches to bring its mouth tip AS CLOSE AS POSSIBLE to
the ground WITHOUT touching it (correctly oriented, mouth pointing down),
then returns to a clean standing pose — all while remaining stable and
robust to pushes. The obs/action spaces are identical to the walking policy
so the two can be switched at runtime with a single key-press.

No structural task-space "down" pose exists: ``mouth_ground_proximity``
pulls the mouth toward the ground, ``head_impact_penalty`` (strong)
forbids ground contact -> the equilibrium is the mouth just above the
ground; ``mouth_perpendicular_to_ground`` orients it downward.

Phase encoding (in the command slot, 3D):
    command = [cos(2*pi*phase), sin(2*pi*phase), 0]
    phase in [0, 0.5)  -> approach (reward mouth going down)
    phase in [0.5, 1)  -> return   (reward returning to standing pose)
Phase is randomized per env on episode reset to decorrelate environments and
avoid synchronised oscillations. See ``microduck_flags.py`` for the period
and segment boundaries.

mjlab 1.3.0 + canonical BAM: fixed (non-accumulating) CoM / head-CoM /
mass-inertia / friction / armature DR, obs-level IMU misalignment,
encoder-bias, obs normalization -- matched to the velocity env, except the
task-specific REGULARIZATION is deliberately kept HEAVIER (slow careful
reaching wants more damping than walking).
"""

from dataclasses import replace

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.tasks.velocity.locomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from mjlab_microduck.tasks.velocity.microduck_scene_cfg import MICRODUCK_ROUGH_TERRAINS_CFG
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_flags import VELOCITY_PUSH_PLAY_INTERVAL_S
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import FEET_GROUND_CFG, HEAD_IMPACT_CFG, MICRODUCK_GROUND_PICK_ROBOT_CFG, SELF_COLLISION_CFG
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckGroundPickFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Microduck ground_pick task on a flat ground plane."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    def __post_init__(self):
        self.scene.entities = {"robot": MICRODUCK_GROUND_PICK_ROBOT_CFG}
        # No terrain-height sensor on this task (even on rough terrain, see
        # MicroduckGroundPickRoughEnvCfg) -- just feet/self-collision/head-impact.
        self.scene.sensors = (FEET_GROUND_CFG, SELF_COLLISION_CFG, HEAD_IMPACT_CFG)
        self.scene.terrain = TerrainEntityCfg(terrain_type="plane")
        self.viewer.body_name = "trunk_base"

        self.actions.joint_pos.scale = 1.0
        # No NeckOffsetJointPositionAction -- head joints are part of the task motion.

        self.curriculum.terrain_levels = None


@configclass
class MicroduckGroundPickRoughEnvCfg(MicroduckGroundPickFlatEnvCfg):
    """Microduck ground_pick task on gentle procedural rough terrain.

    Note: unlike ``velocity``'s rough variant, this doesn't add contact
    softening / higher solver iterations / raised ``nconmax`` -- the
    original file didn't either (the ground_pick motion is quasi-static, not
    a dynamic gait, so the rough-terrain contact-solver stress that
    motivated those velocity-specific tweaks doesn't apply here).
    """

    def __post_init__(self):
        super().__post_init__()

        # replace(...) (not the bare module-level constant): avoids aliasing
        # MICRODUCK_ROUGH_TERRAINS_CFG across every env cfg built from it --
        # see tasks/velocity/microduck_velocity_env_cfg.py's identical note.
        self.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(MICRODUCK_ROUGH_TERRAINS_CFG),
        )
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel, params={"command_name": "twist"})


@configclass
class MicroduckGroundPickFlatEnvCfg_PLAY(MicroduckGroundPickFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Spaced-out push interval in play mode, for judging the motion on
        # realistic behavior rather than a stress test.
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S


@configclass
class MicroduckGroundPickRoughEnvCfg_PLAY(MicroduckGroundPickRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S

        assert self.scene.terrain is not None
        assert self.scene.terrain.terrain_generator is not None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5


def make_microduck_ground_pick_env_cfg(play: bool = False, rough: bool = False):
    """Create the Microduck ground_pick environment configuration.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration keeps working unmodified.
    """
    cfg_cls = {
        (False, False): MicroduckGroundPickFlatEnvCfg,
        (True, False): MicroduckGroundPickFlatEnvCfg_PLAY,
        (False, True): MicroduckGroundPickRoughEnvCfg,
        (True, True): MicroduckGroundPickRoughEnvCfg_PLAY,
    }[(play, rough)]
    return cfg_cls().to_mjlab_cfg()
