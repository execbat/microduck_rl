"""Termination specifications for the Microduck roller_standup task.

The robot STARTS fallen -- a tilt-based termination would kill the episode
on the first step. ``nan_state`` (inherited unchanged from
``velocity_rollers``) stays.
"""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks.velocity_rollers.microduck_terminations_cfg import (
    MicroduckTerminationsCfg as _RollersTerminationsCfg,
)
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckTerminationsCfg(_RollersTerminationsCfg):
    fell_over: DoneTerm | None = None
