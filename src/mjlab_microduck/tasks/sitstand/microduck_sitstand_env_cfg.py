"""Microduck sitstand task — commanded sit <-> stand, GENTLY.

One policy, both directions, driven by a posture command:
    cmd (twist slot) = [sit_flag, 0, 0]   sit_flag in {0 = STAND, 1 = SIT}
"Stand" is the all-zero command, the same deployment idle as every other
policy. The command flips mid-episode with a dwell time of a few seconds, so
each episode trains descents, seated rest, rises and standing rest, plus
"hold what you're already doing" (reset state x command are independent).

Design synthesis:
  - Posture-conditioned single-target rewards (``posture_*`` in
    ``microduck_rewards_cfg.py``): the target (SIT keyframe + SIT_Z vs HOME +
    STAND_Z) is selected per env from the live command. No trajectory, no
    waypoints, no phase timing -- the policy discovers its own transition
    path (knee-down first, head assist, etc. are all allowed: full-collision
    model, no head-ground penalty, no fall termination).
  - Gentleness both ways: descent-speed cap (proven recipe, -10 from step 0)
    AND a mirrored rise-speed cap (introduced by curriculum AFTER the rise is
    discovered -- the standup attempt-tax lesson), plus the |a_z| shock
    penalty throughout.
  - Rest quality: ``posture_stillness`` (velocity-Gaussian at the commanded
    height, tilt-gated) + ``posture_composite`` (multiplicative
    height*upright*pose vs the commanded target -- partial-sum exploits
    like plank/flop/lean collapse to ~0).
  - Head commandable in BOTH postures (head_pose command + tracking, exactly
    like velocity/standup), body_command slot zero-padded -> 61D obs parity.
  - Sim2real: velocity-parity DR / obs noise / delays / regularisers (the
    transferring recipe), contact-solver hardening (nconmax=200, iters
    30/50 -- seated contact NaN fix), delayed push ramp (pushes early made
    the policy unlearn sitting).

Keyframes (stability-verified, keep in sync with the standup env):
  SIT  = knee +-1.35, hip_pitch -+0.4079, ankle/hip_roll 0, trunk z 0.060
         (swept 2026-07-27 -- the old keyframe tipped over; verify TILT in
         sim before changing this pose).
  STAND = HOME joints, trunk z 0.115 (measured standing equilibrium).

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

from dataclasses import replace

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.tasks.locomotion.velocity import LocomotionVelocityFlatEnvCfg
from mjlab_microduck.tasks.velocity.microduck_scene_cfg import MICRODUCK_ROUGH_TERRAINS_CFG
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_flags import EPISODE_LENGTH_S, SIM_ITERATIONS, SIM_LS_ITERATIONS, SIM_NCONMAX, VELOCITY_PUSH_PLAY_INTERVAL_S
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import FEET_GROUND_CFG, MICRODUCK_STANDUP_ROBOT_CFG, SELF_COLLISION_CFG
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckSitStandFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    """Microduck sitstand task on a flat ground plane."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    episode_length_s: float = EPISODE_LENGTH_S

    def __post_init__(self):
        # Standup robot variant: full collision meshes -- the body must
        # physically rest on the ground while seated, and knees/head may
        # touch mid-transition.
        self.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
        self.scene.sensors = (FEET_GROUND_CFG, SELF_COLLISION_CFG)
        self.scene.terrain = TerrainEntityCfg(terrain_type="plane")
        self.viewer.body_name = "trunk_base"

        self.actions.joint_pos.scale = 1.0

        # MuJoCo physics robustness (contact NaN fix), UNCONDITIONAL --
        # applies to both Flat and Rough (unlike velocity's rough-only
        # tuning): the seated pose alone (regardless of terrain) puts trunk
        # + folded legs + head all in close ground/self contact and
        # overflows the default contact solver. See microduck_flags.py.
        self.sim.nconmax = SIM_NCONMAX
        self.sim.mujoco.iterations = SIM_ITERATIONS
        self.sim.mujoco.ls_iterations = SIM_LS_ITERATIONS

        self.curriculum.terrain_levels = None


@configclass
class MicroduckSitStandRoughEnvCfg(MicroduckSitStandFlatEnvCfg):
    """Microduck sitstand task on gentle procedural rough terrain."""

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
class MicroduckSitStandFlatEnvCfg_PLAY(MicroduckSitStandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S


@configclass
class MicroduckSitStandRoughEnvCfg_PLAY(MicroduckSitStandRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S

        assert self.scene.terrain is not None
        assert self.scene.terrain.terrain_generator is not None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5


def make_microduck_sitstand_env_cfg(play: bool = False, rough: bool = False):
    """Create the Microduck sitstand environment configuration.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration keeps working unmodified.
    """
    cfg_cls = {
        (False, False): MicroduckSitStandFlatEnvCfg,
        (True, False): MicroduckSitStandFlatEnvCfg_PLAY,
        (False, True): MicroduckSitStandRoughEnvCfg,
        (True, True): MicroduckSitStandRoughEnvCfg_PLAY,
    }[(play, rough)]
    return cfg_cls().to_mjlab_cfg()
