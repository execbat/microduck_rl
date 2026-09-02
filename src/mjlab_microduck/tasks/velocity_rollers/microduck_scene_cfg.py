"""Scene wiring for the Microduck velocity_rollers task."""

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG

__all__ = ["MICRODUCK_WALK_ROLLERS_ROBOT_CFG", "FEET_GROUND_CFG", "SELF_COLLISION_CFG"]

# 2026-07 model: the roller_blade bodies were merged into the ankles (blade
# mesh is now a visual geom on ankle_{l,r}_v1); the tires hang directly off
# the ankles. Each ankle subtree's only collision geoms are its two tires,
# so this keeps the old per-foot semantics: 2 slots, left first.
FEET_GROUND_CFG = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(ankle_l_v1|ankle_r_v1)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)

SELF_COLLISION_CFG = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)
