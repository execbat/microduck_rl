"""Scene wiring for the Microduck velocity task: robot, sensors, terrain.

Split out of the old monolithic ``microduck_velocity_env_cfg.py`` so the
sensor/terrain definitions can be read (and reused by other Microduck tasks)
independently of rewards/observations/etc.
"""

import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG

__all__ = [
    "MICRODUCK_WALK_ROBOT_CFG",
    "MICRODUCK_ROUGH_TERRAINS_CFG",
    "SITE_NAMES",
    "FOOT_FRICTION_GEOM_NAMES",
    "feet_ground_cfg",
    "self_collision_cfg",
    "foot_height_scan_cfg",
    "soften_terrain_contacts",
]

SITE_NAMES = ("left_foot", "right_foot")
FOOT_FRICTION_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")

# Contact sensor for feet - LEFT, RIGHT order.
feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="geom",
        pattern=r"^(left_foot_collision|right_foot_collision)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)

self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)

# mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height rewards are
# driven by a per-foot terrain-height ray sensor (was site_pos based). Mirrors
# microban's foot_height_scan.
foot_height_scan_cfg = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES),
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
)

# Microduck-specific rough terrain: much gentler than the default ROUGH_TERRAINS_CFG.
# The robot can only lift its feet ~1-2 cm, so steps are capped at 1.5 cm.
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=2, #10,
    num_cols=2, #20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),  # max 1.5 cm (vs 10 cm default)
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # NOTE: BoxInvertedPyramidStairsTerrainCfg removed -- it sets env_origin_z to the pit
        # bottom (negative), causing resets at root_z = 0.12 + env_origin_z ~ -0.10 m which
        # places the robot below the pit floor and makes it fall through the ground.
        # Uneven cobblestone-like ground: random per-cell height offsets.
        # grid_width=0.12 on an 8m patch = 66x66 = 4 356 boxes/patch -> ~261 K total -> OOM.
        # 0.45 m gives 17x17 = 289 boxes/patch -> ~17 K total (border = 0.35 m OK).
        # Must not divide evenly into terrain size (8.0 m): 0.45 x 17 = 7.65 OK
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # max 1 cm
            platform_width=1.5,
        ),
        # Gentle slopes (heightfield pyramid, platform on TOP -- robot spawns on
        # the flat platform and walks down/up/across the slope as commands
        # resample). slope_range is rise/run: 0.03->0.10 ~ 1.7deg->5.7deg by
        # difficulty -- small robot, small slopes. NOT inverted (see the
        # inverted-pyramid env_origin note above -- same pit-spawn risk class).
        # vertical_scale=0.001 keeps quantization steps at 1 mm so a gentle
        # slope is smooth instead of a staircase of 5 mm ledges.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """Soften terrain box geom contacts to reduce edge-contact NaN instability.

    Box terrains place adjacent geoms at different heights. The hard edges where
    heights change cause contact normal instability when feet land on them, which
    can produce impulsive NaN forces in the MuJoCo solver.

    Doubling the solref time constant (0.02 -> 0.04 s) makes contact springs
    2x softer -- enough to damp the instability without noticeably changing the
    macro-level walking physics. Applied to all geoms in the "terrain" body,
    which contains every box generated by TerrainGenerator.
    """
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]  # 2x softer time constant (default: 0.02)
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # slightly softer impedance
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")
