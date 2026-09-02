"""Reusable velocity locomotion bases with explicit terrain selection.

``LocomotionVelocityEnvCfg`` supplies common managers and simulation settings
without choosing terrain. The flat and rough subclasses choose their scene;
only the rough base enables the body terrain scanner and terrain curriculum.
Robot assets, sensor frames, and robot-specific terms belong to concrete tasks.
``to_mjlab_cfg()`` converts declarative configs into mjlab's manager dictionaries.
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

from .cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventsCfg,
    ObservationsCfg,
    RewardsCfg,
    RoughCurriculumCfg,
    RoughObservationsCfg,
    TerminationsCfg,
)

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
class LocomotionVelocityEnvCfg:
    """Terrain-neutral velocity template; concrete tasks supply the robot."""

    # -- Managers (one component per file under cfg/) ---------------------
    scene: SceneCfg = SceneCfg(
        terrain=None,
        sensors=(FOOT_HEIGHT_SCAN_SENSOR,),
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


@configclass
class LocomotionVelocityFlatEnvCfg(LocomotionVelocityEnvCfg):
    """Velocity template on a plane, without body terrain observations."""

    scene: SceneCfg = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        sensors=(FOOT_HEIGHT_SCAN_SENSOR,),
        num_envs=1,
        extent=2.0,
    )


@configclass
class LocomotionVelocityRoughEnvCfg(LocomotionVelocityEnvCfg):
    """Velocity template with procedural terrain, height scans and curriculum."""

    # configclass deep-copies the entire scene (including the generator and
    # sensors) per instance, so play-mode edits cannot change other configs.
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
    observations: RoughObservationsCfg = RoughObservationsCfg()
    curriculum: RoughCurriculumCfg = RoughCurriculumCfg()
