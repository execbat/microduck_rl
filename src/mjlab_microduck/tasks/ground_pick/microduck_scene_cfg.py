"""Scene wiring for the Microduck ground_pick task."""

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUND_PICK_ROBOT_CFG

__all__ = [
    "MICRODUCK_GROUND_PICK_ROBOT_CFG",
    "FEET_GROUND_CFG",
    "SELF_COLLISION_CFG",
    "HEAD_IMPACT_CFG",
    "FOOT_FRICTION_GEOM_NAMES",
]

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

# Head-on-ground impact sensor -- covers the neck subtree (head_plate,
# head_shell, etc). Used by head_impact_penalty to discourage the policy
# from slamming the head into the ground during the approach.
HEAD_IMPACT_CFG = ContactSensorCfg(
    name="head_impact_contact",
    primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
)
