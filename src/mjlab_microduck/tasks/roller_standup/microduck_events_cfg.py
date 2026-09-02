"""Event specifications for the Microduck roller_standup task."""

import math

from mjlab.managers.event_manager import EventTermCfg as EventTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity_rollers.microduck_events_cfg import MicroduckEventsCfg as _RollersEventsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckEventsCfg(_RollersEventsCfg):
    """Event terms for the Microduck roller_standup task.

    Every field inherited from ``velocity_rollers`` (DR, push, friction, ...)
    is kept unchanged -- ``set_ground_state`` is the only new field, and it
    matters that it's declared LAST: event execution order follows
    declaration order (see ``ball_kick``'s ``microduck_events_cfg.py``
    docstring for the general mechanism), and this term must overwrite the
    pose ``reset_base``/``reset_robot_joints`` set.

    No "sitting" bucket (unlike ``standup``): there's no roller equivalent
    of the sit-policy hand-off, and standup's ``sitting_joint_overrides`` are
    indices for the WHEEL-LESS model anyway.
    """

    set_ground_state: EventTerm | None = EventTerm(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,  # belly (+90deg pitch)
            "face_up_prob": 0.00,  # back -- hardest, introduced late
            "sitting_prob": 0.00,
            "standing_prob": 0.50,
            "sitting_joint_overrides": None,
            # The two ground poses (belly/back) share a SINGLE z range, but
            # their contacts have nothing in common: the belly only lifts
            # off the ground from 0.0752, the back rests at 0.0475. A
            # single floor can't be ideal for both. 0.076 is chosen to
            # eliminate any belly-side interpenetration (measured: at 0.05,
            # +25mm into the ground), at the cost of a back-start 28-42mm
            # above its rest -- a much gentler artifact than a contact
            # pushout.
            "prone_z_min": 0.076,
            "prone_z_max": 0.09,
            # Standing on rollers: ROLLER_STAND_Z = 0.138 (vs. 0.11-0.12
            # without wheels).
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # Pitch/roll noise at spawn. Note: in set_random_ground_state
            # the "standing" bucket reuses the "sitting" bucket's
            # quaternion, so this noise ALSO applies to standing starts --
            # intentional (no over-fitting to perfectly upright).
            "sitting_tilt_max": math.radians(10),
        },
    )
