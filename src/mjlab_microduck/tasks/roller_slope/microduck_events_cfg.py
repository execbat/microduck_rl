"""Event specifications for the Microduck roller_slope task.

``reset_base``'s ``velocity_range`` is already empty on the inherited
``velocity_rollers`` base -- only ``yaw`` actually changes here (random
-> fixed, facing downhill). No base push is injected: the robot spawns on
the ramp (see ``SPAWN_ON_RAMP``) and gravity rolls the wheels (momentum
delivered to the wheels, no slip). The old approach (a base-only push,
wheels stationary) skidded -> contact spike -> NaN divergence, and the
robot "walked to stop" instead of rolling.
"""

from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roller_slope.microduck_flags import ENTRY_VELOCITY_X, SPAWN_YAW
from mjlab_microduck.tasks.velocity_rollers.microduck_events_cfg import MicroduckEventsCfg as _RollersEventsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckEventsCfg(_RollersEventsCfg):
    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),  # inherited unchanged.
                "y": (-0.5, 0.5),  # inherited unchanged.
                "z": (0.1335, 0.1435),  # inherited unchanged.
                "yaw": SPAWN_YAW,  # fixed facing downhill (was random, (-3.14, 3.14)).
            },
            "velocity_range": {},  # inherited unchanged.
        },
    )
    # reset_action_history is inherited unchanged from velocity_rollers
    # (same func/params) -- the original file reassigned it too, but to an
    # identical value, so nothing to override here.

    # -- New event term (appended after all inherited fields above) --
    # Rolling-entry momentum (wheel spin-up in sync with base velocity, no
    # slip). AFTER reset_base (guaranteed: this is a new field, appended
    # after every inherited field -- see ball_kick's microduck_events_cfg.py
    # docstring for the general field-order mechanism).
    reset_rolling_entry: EventTerm | None = EventTerm(
        func=microduck_mdp.reset_rolling_entry,
        mode="reset",
        params={"speed_range": ENTRY_VELOCITY_X},
    )
