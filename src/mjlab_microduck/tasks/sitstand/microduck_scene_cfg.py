"""Scene wiring for the Microduck sitstand task.

NOTE: no head-ground contact penalty here (unlike ``roulade``/``standup``'s
older sit env). Using the head as a third support point during transitions
is explicitly allowed -- the plank-as-terminal-rest exploit is anti-selected
by ``posture_composite``/``posture_stillness`` instead (both ~=0 at plank
tilt/height).
"""

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
