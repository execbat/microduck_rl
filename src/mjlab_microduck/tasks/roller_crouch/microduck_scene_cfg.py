"""Scene wiring for the Microduck roller_crouch task.

Same robot and sensor set as ``tasks/velocity_rollers`` -- this task rides
on the roller physics, layering a one-shot crouch-glide trick on top.
"""

from mjlab_microduck.tasks.velocity_rollers.microduck_scene_cfg import (
    FEET_GROUND_CFG,
    MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
    SELF_COLLISION_CFG,
)

__all__ = ["MICRODUCK_WALK_ROLLERS_ROBOT_CFG", "FEET_GROUND_CFG", "SELF_COLLISION_CFG"]
