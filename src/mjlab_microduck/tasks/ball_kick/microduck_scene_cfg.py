"""Scene wiring for the Microduck BallKick task: robot, ball, sensors.

The support-foot contact sensor depends on which foot is kicking, so unlike
the velocity task's static module-level sensor objects, this file exposes a
small factory (``build_sensors``) that the env cfg calls in
``__post_init__`` once it knows ``self.kick_foot``.
"""

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_CFG, MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks.ball_kick.microduck_flags import support_foot_of

__all__ = [
    "MICRODUCK_STANDUP_ROBOT_CFG",
    "MICRODUCK_BALL_CFG",
    "FOOT_FRICTION_GEOM_NAMES",
    "build_sensors",
]

FOOT_FRICTION_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")


def build_sensors(kick_foot: str) -> tuple[ContactSensorCfg, ContactSensorCfg, ContactSensorCfg]:
    """Feet/support-foot/self-collision contact sensors for the given kicking foot."""
    support_foot = support_foot_of(kick_foot)

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

    # Support-foot sensor: the non-kicking foot must stay planted through the kick.
    support_foot_ground_cfg = ContactSensorCfg(
        name="support_foot_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=rf"^{support_foot}_foot_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    return feet_ground_cfg, support_foot_ground_cfg, self_collision_cfg
