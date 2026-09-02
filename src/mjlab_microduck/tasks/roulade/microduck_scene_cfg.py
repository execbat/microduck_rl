"""Scene wiring for the Microduck roulade task."""

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

__all__ = [
    "MICRODUCK_STANDUP_ROBOT_CFG",
    "FEET_GROUND_CFG",
    "SELF_COLLISION_CFG",
    "HEAD_GROUND_CFG",
    "ROBOT_GROUND_CFG",
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

# Head-ground contact -- the roll's pivot signal. jaw_soft is the body that
# carries the head collision geoms (top_head_shell = the flat top, jaw,
# bottom_head_shell) in robot_allcollisions.xml. NAME IS LOAD-BEARING:
# _update_roulade_accum (in mdp/_common.py) reads it for the
# over-the-head latch.
HEAD_GROUND_CFG = ContactSensorCfg(
    name="head_ground_contact",
    primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)

# Whole-robot ground contact -- the SUPPORT GATE: the rotation accumulator
# only integrates while some robot geom touches the terrain, so ballistic
# flips ("breakdance") earn no progress and never complete. NAME IS
# LOAD-BEARING: _update_roulade_accum reads it.
ROBOT_GROUND_CFG = ContactSensorCfg(
    name="robot_ground_contact",
    primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)
