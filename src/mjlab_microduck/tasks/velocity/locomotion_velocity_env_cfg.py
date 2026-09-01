"""Generic ("rough terrain") velocity locomotion task, IsaacLab-style.

This mirrors the structure of ``isaaclab_tasks.manager_based.locomotion
.velocity.velocity_env_cfg.LocomotionVelocityRoughEnvCfg``: one top-level
``@configclass`` whose fields are the individual manager configs, each of
which lives in its own file under ``cfg/``. A robot-specific task (see
``microduck_velocity_env_cfg.py`` next to this file) subclasses this and
overrides/extends fields, exactly like ``G1RoughEnvCfg(LocomotionVelocity
RoughEnvCfg)`` does for Isaac Lab's own G1.

The one thing that differs from "real" IsaacLab: the *runtime* here is mjlab,
not Isaac Sim, and mjlab's managers want plain dicts rather than attribute
containers. ``to_mjlab_cfg()`` is the single place that bridges the two --
every other file in this package only ever deals with the declarative,
attribute-style configs.
"""

from dataclasses import field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.viewer import ViewerConfig

from mjlab_microduck.utils.configclass import configclass
from mjlab_microduck.utils.manager_compat import group_to_dict, observations_to_dict

from .cfg import ActionsCfg, CommandsCfg, CurriculumCfg, EventsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg

# Sensor names referenced by the base observation/reward terms (see cfg/*.py).
# Frames are left unset here ("" / ()) -- a robot cfg must wire them to real
# body/site names, same as mjlab's own make_velocity_env_cfg().
TERRAIN_SCAN_SENSOR = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
)

FOOT_HEIGHT_SCAN_SENSOR = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
        show_rays=True,
        hit_color=(1.0, 0.0, 1.0, 0.8),
        hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
)


@configclass
class LocomotionVelocityRoughEnvCfg:
    """Generic rough-terrain velocity-tracking locomotion task."""

    # -- Managers (one component per file under cfg/) ---------------------
    scene: SceneCfg = SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
        ),
        sensors=(TERRAIN_SCAN_SENSOR, FOOT_HEIGHT_SCAN_SENSOR),
        num_envs=1,
        extent=2.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # -- Base env settings --------------------------------------------------
    decimation: int = 4
    episode_length_s: float = 20.0
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            nconmax=35,
            njmax=1500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        )
    )
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        )
    )

    def to_mjlab_cfg(self) -> ManagerBasedRlEnvCfg:
        """Convert this declarative config tree into mjlab's native, dict-based
        ``ManagerBasedRlEnvCfg`` -- the object mjlab's env, train/play scripts,
        and gym registration actually consume.
        """
        return ManagerBasedRlEnvCfg(
            decimation=self.decimation,
            scene=self.scene,
            observations=observations_to_dict(self.observations),
            actions=group_to_dict(self.actions),
            commands=group_to_dict(self.commands),
            events=group_to_dict(self.events),
            rewards=group_to_dict(self.rewards),
            terminations=group_to_dict(self.terminations),
            curriculum=group_to_dict(self.curriculum),
            sim=self.sim,
            viewer=self.viewer,
            episode_length_s=self.episode_length_s,
        )
