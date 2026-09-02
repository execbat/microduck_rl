"""Scene wiring for the Microduck roller_slope task.

Robot/sensors are unchanged from ``tasks/velocity_rollers`` (re-exported by
the env cfg from there directly) -- only the terrain differs, so this file
only holds the terrain factory.
"""

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks.roller_slope.microduck_flags import (
    FLAT_LENGTH,
    RAMP_LENGTH_RANGE,
    RUNOUT_LENGTH,
    SPAWN_ON_RAMP,
    TILE_SIZE,
    resolve_play_difficulty,
)
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg

__all__ = ["build_terrain"]


def build_terrain(play: bool) -> TerrainEntityCfg:
    """Flat+ramp+runout terrain, curriculum-driven stiffness.

    In play mode, shows varied slopes: difficulty ``None`` -> random
    stiffness per env (level drawn across every row); a 0..1 value forces a
    specific stiffness (1.0 = steepest). Controlled via
    ``SLOPE_PLAY_DIFFICULTY`` (see ``microduck_flags.py``).
    """
    terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TILE_SIZE,
            curriculum=True,
            num_rows=10,  # 10 stiffness levels
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "flat_ramp": FlatRampTerrainCfg(
                    flat_length=FLAT_LENGTH,
                    ramp_length_range=RAMP_LENGTH_RANGE,
                    runout_length=RUNOUT_LENGTH,
                    spawn_on_ramp=SPAWN_ON_RAMP,
                )
            },
        ),
        max_init_terrain_level=0,  # curriculum: start on the gentlest ramp
    )
    if play:
        play_difficulty = resolve_play_difficulty()
        if play_difficulty is not None:
            terrain.terrain_generator.difficulty_range = (play_difficulty, play_difficulty)
        else:
            terrain.max_init_terrain_level = None
    return terrain
