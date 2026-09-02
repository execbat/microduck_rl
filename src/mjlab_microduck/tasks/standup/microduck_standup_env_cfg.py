"""Microduck standup task — sitting/prone pose -> standing, then commandable.

Episodic policy that gently rises from a sitting keyframe (or a prone
face-down/face-up recovery pose) to the standing keyframe. Companion to
``sitstand``'s sit direction -- together they form a clean sit<->stand pair.

Reset: sitting keyframe (trunk z ~= 0.06) OR one of 3 other ground states
(standing, face-down, face-up) -- see ``microduck_events_cfg.py``'s
``set_ground_state``, ramped easy->hard by the ``ground_state_mix``
curriculum.
Target: standing keyframe (trunk z ~= 0.115, HOME joints).
Reward design: a single fixed target is rewarded from t=0 to end of
episode; gentleness is enforced via |a_z| only; smoothness is enforced by
the usual sim2real regularisers. No trajectory waypoints, no
episode-progress gating -- the policy is free to discover its own rise path.

Body control (see ``ENABLE_BODY_CONTROL`` in ``microduck_flags.py``): once
standing, the policy tracks a commanded trunk delta [z, roll, pitch] from
the nominal stand (the real ``body_pose`` command in the previously
zero-padded 6D obs slot). Kicks in at iter 2500 via the body-control
curricula, after the ``ground_state_mix`` recovery curriculum has finished
ramping.
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
from .microduck_flags import EPISODE_LENGTH_S, VELOCITY_PUSH_PLAY_INTERVAL_S
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import FEET_GROUND_CFG, MICRODUCK_STANDUP_ROBOT_CFG, SELF_COLLISION_CFG
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckStandupFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    """Microduck standup task on a flat ground plane."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    episode_length_s: float = EPISODE_LENGTH_S

    def __post_init__(self):
        self.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
        self.scene.sensors = (FEET_GROUND_CFG, SELF_COLLISION_CFG)
        self.scene.terrain = TerrainEntityCfg(terrain_type="plane")
        self.viewer.body_name = "trunk_base"

        self.actions.joint_pos.scale = 1.0

        self.curriculum.terrain_levels = None


@configclass
class MicroduckStandupRoughEnvCfg(MicroduckStandupFlatEnvCfg):
    """Microduck standup task on gentle procedural rough terrain."""

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

        # Contact-solver hardening for rough terrain (same fix as velocity's
        # rough variant, and as sitstand's -- both for the same underlying
        # reason). The default nconmax=35 / 10 solver iterations are tight
        # even on flat ground for this full-collision robot; add rough-
        # terrain box contacts on top of a task that spends most of its
        # time falling/flipping (prone recovery, sit->stand transitions)
        # and the contact solver overflows -> silently dropped contacts ->
        # sudden decompression -> NaN / crash. Not present in the original
        # (pre-refactor) file -- this task combination was apparently never
        # actually run on rough terrain before.
        self.sim.nconmax = 200
        self.sim.mujoco.iterations = 30
        self.sim.mujoco.ls_iterations = 50


@configclass
class MicroduckStandupFlatEnvCfg_PLAY(MicroduckStandupFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S


@configclass
class MicroduckStandupRoughEnvCfg_PLAY(MicroduckStandupRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S

        assert self.scene.terrain is not None
        assert self.scene.terrain.terrain_generator is not None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5


def make_microduck_standup_env_cfg(play: bool = False, rough: bool = False):
    """Create the Microduck standup environment configuration.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration keeps working unmodified.
    """
    cfg_cls = {
        (False, False): MicroduckStandupFlatEnvCfg,
        (True, False): MicroduckStandupFlatEnvCfg_PLAY,
        (False, True): MicroduckStandupRoughEnvCfg,
        (True, True): MicroduckStandupRoughEnvCfg_PLAY,
    }[(play, rough)]
    return cfg_cls().to_mjlab_cfg()
