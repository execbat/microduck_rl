"""Termination specifications for the Microduck roller_slope task.

The runout at the bottom of the ramp gives solid ground, so no need to
terminate "at the edge" anymore (``out_of_terrain_bounds`` used to cut long
ramps short too early -- dropped). Kept: falling (``bad_orientation``), NaN
(inherited unchanged from ``velocity_rollers``), and "fell into the void"
(trunk below the lowest runout) in case the robot leaves solid ground.
"""

from mjlab.envs import mdp as base_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roller_slope.microduck_flags import VOID_FLOOR
from mjlab_microduck.tasks.velocity_rollers.microduck_terminations_cfg import (
    MicroduckTerminationsCfg as _RollersTerminationsCfg,
)
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckTerminationsCfg(_RollersTerminationsCfg):
    # nan_state is inherited unchanged from velocity_rollers (same func/params).
    out_of_terrain_bounds: DoneTerm | None = None
    fell_over: DoneTerm | None = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    fell_into_void: DoneTerm | None = DoneTerm(
        func=microduck_mdp.root_height_below,
        params={"min_height": VOID_FLOOR, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
