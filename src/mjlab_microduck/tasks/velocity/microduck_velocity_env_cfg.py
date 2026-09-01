"""Microduck velocity (walking) environment -- top-level assembly.

This is the direct successor of the old ~950-line
``tasks/microduck_velocity_env_cfg.py``. All of that file's content has been
split by manager into sibling files in this package (``microduck_observations
_cfg.py``, ``microduck_rewards_cfg.py``, ``microduck_commands_cfg.py``,
``microduck_events_cfg.py``, ``microduck_terminations_cfg.py``,
``microduck_curriculum_cfg.py``, ``microduck_scene_cfg.py``,
``microduck_flags.py``); this file only assembles them into env cfg classes,
mirroring IsaacLab's own
``G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg)`` / ``G1RoughEnvCfg_PLAY`` /
``G1FlatEnvCfg`` split.

``make_microduck_velocity_env_cfg(play=..., rough=...)`` keeps the exact same
signature and return type (``mjlab.envs.ManagerBasedRlEnvCfg``) as before, so
``tasks/__init__.py`` (gym registration) and ``tasks/backlash.py``
(``make_backlash_variant``) need no changes.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.utils.configclass import configclass

from .locomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_flags import VELOCITY_PUSH_PLAY_INTERVAL_S
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    MICRODUCK_WALK_ROBOT_CFG,
    feet_ground_cfg,
    self_collision_cfg,
    foot_height_scan_cfg,
    soften_terrain_contacts,
)
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckVelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Microduck velocity task on a flat ground plane."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    def __post_init__(self):
        self.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
        # Drop the generic terrain_scan placeholder sensor (unused -- this
        # robot has no body-mounted terrain raycaster) and wire in the
        # robot-specific foot/self-collision contact sensors instead.
        self.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
        self.scene.terrain = TerrainEntityCfg(terrain_type="plane")

        self.viewer.body_name = "trunk_base"
        self.actions.joint_pos.scale = 1.0


@configclass
class MicroduckVelocityRoughEnvCfg(MicroduckVelocityFlatEnvCfg):
    """Microduck velocity task on gentle procedural rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        # `replace(...)` (not the bare module-level constant): the play-mode
        # subclass below mutates `.terrain_generator` in place (num_rows,
        # curriculum, ...). Without a fresh copy here, that mutation would
        # alias back into MICRODUCK_ROUGH_TERRAINS_CFG and leak into every
        # other env cfg built from it -- a real bug in the pre-refactor code,
        # where task registration order in tasks/__init__.py happened to
        # mask it. mjlab's own velocity_env_cfg.py uses the same
        # `replace(ROUGH_TERRAINS_CFG)` pattern for exactly this reason.
        self.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(MICRODUCK_ROUGH_TERRAINS_CFG),
        )
        # Soften terrain box contacts: adjacent boxes at different heights
        # create hard edges that destabilise the contact solver and produce
        # NaN forces.
        self.scene.spec_fn = soften_terrain_contacts

        # The velocity env default nconmax=35 is tight for rough terrain: when
        # the robot falls and multiple body links hit multiple boxes
        # simultaneously, contacts overflow -> some are silently dropped ->
        # sudden decompression -> NaN.
        self.sim.nconmax = 200  # was 35
        # The velocity env uses only 10 solver iterations (vs the default
        # 100), too few to resolve edge contacts on rough box terrain.
        # Tripling iterations significantly reduces contact resolution
        # failures with a modest compute cost on GPU (MJWarp parallelises
        # across envs).
        self.sim.mujoco.iterations = 30  # was 10
        self.sim.mujoco.ls_iterations = 50  # was 20

        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel, params={"command_name": "twist"}
        )


@configclass
class MicroduckVelocityFlatEnvCfg_PLAY(MicroduckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Shorter push interval in play mode, for better visibility.
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S


@configclass
class MicroduckVelocityRoughEnvCfg_PLAY(MicroduckVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S

        assert self.scene.terrain is not None
        assert self.scene.terrain.terrain_generator is not None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5


def make_microduck_velocity_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck velocity tracking environment configuration.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration and ``make_backlash_variant``
    keep working unmodified.
    """
    cfg_cls = {
        (False, False): MicroduckVelocityFlatEnvCfg,
        (True, False): MicroduckVelocityFlatEnvCfg_PLAY,
        (False, True): MicroduckVelocityRoughEnvCfg,
        (True, True): MicroduckVelocityRoughEnvCfg_PLAY,
    }[(play, rough)]
    return cfg_cls().to_mjlab_cfg()
