"""Scene wiring for the Microduck standup task."""

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

__all__ = ["MICRODUCK_STANDUP_ROBOT_CFG", "FEET_GROUND_CFG", "SELF_COLLISION_CFG", "FOOT_FRICTION_GEOM_NAMES"]

FOOT_FRICTION_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")

FEET_GROUND_CFG = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
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
