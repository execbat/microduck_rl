"""Event specifications for the Microduck velstand task.

Subclasses ``tasks.velocity``'s own ``MicroduckEventsCfg`` -- adds a single
new event (``random_prone_init``), ramped in by the ``prone_init_prob``
curriculum. Everything else (DR, push, friction, ...) flows in from the
walk layer unchanged.
"""

from mjlab.managers.event_manager import EventTermCfg as EventTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.microduck_events_cfg import MicroduckEventsCfg as _VelocityEventsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckEventsCfg(_VelocityEventsCfg):
    # z fix (audit BUG in the pre-rebase design): the function defaults
    # were 0.20-0.25m -- a 15-20cm free-fall opening every prone episode.
    # Face-down trunk rests at ~0.044m; spawn just above the ground instead.
    random_prone_init: EventTerm | None = EventTerm(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,  # ramped by the prone_init_prob curriculum
            "face_down_prob": 1.0,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,  # ramped by the prone_init_prob curriculum
        },
    )
