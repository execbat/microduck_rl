"""Tunables for the Microduck roller_slope task.

Direct port of the module-level constants from the old
``tasks/microduck_roller_slope_env_cfg.py``. Robot spawns on flat ground
(small forward impulse), rolls down a descending ramp, and glides while
staying balanced -- no steering, the twist command is neutralised
(rel_standing_envs=1.0). Custom flat+ramp terrain (``FlatRampTerrainCfg``),
stiffness curriculum (``terrain_levels_slope``).
"""

import math
import os

from mjlab_microduck.tasks.slope_terrain import RAMP_DEG_MAX

# Flat+ramp+runout terrain geometry.
FLAT_LENGTH = 2.0
RAMP_LENGTH_RANGE = (3.0, 8.0)  # horizontal ramp length, randomised per tile
RUNOUT_LENGTH = 4.0  # flat runout at the bottom
SPAWN_ON_RAMP = 0.3  # spawn this many meters onto the ramp (gravity -> rolling, no slip)
ENTRY_VELOCITY_X = (0.25, 0.45)  # small initial forward/downhill momentum (m/s)
TILE_SIZE = (15.0, 4.0)  # >= flat + ramp_max + runout (= 14) + margin
SPAWN_YAW = (0.0, 0.0)  # facing downhill (+x), fixed

# Play-mode stiffness: None = random (same as training). Set a 0..1 value to
# force a specific slope (1.0 = steepest, ~20deg; 0.5 = mid). Overridable
# without editing code via the SLOPE_PLAY_DIFFICULTY env var (e.g.
# SLOPE_PLAY_DIFFICULTY=1.0 uv run play ... ; "none"/"random" = random).
PLAY_DIFFICULTY = None


def resolve_play_difficulty():
    """Play-mode difficulty: SLOPE_PLAY_DIFFICULTY env var, else the constant."""
    raw = os.environ.get("SLOPE_PLAY_DIFFICULTY")
    if raw is None:
        return PLAY_DIFFICULTY
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_slope] SLOPE_PLAY_DIFFICULTY='{raw}' invalid -- falling back to {PLAY_DIFFICULTY}")
        return PLAY_DIFFICULTY


# "Fell into the void" termination: below the lowest runout (steepest,
# longest ramp), with margin -- never triggers during a normal descent,
# only if the robot leaves the solid ground.
_MAX_DROP = RAMP_LENGTH_RANGE[1] * math.tan(math.radians(RAMP_DEG_MAX))
VOID_FLOOR = -_MAX_DROP - 0.5
