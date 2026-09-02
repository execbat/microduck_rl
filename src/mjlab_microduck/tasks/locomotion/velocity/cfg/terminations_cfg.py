"""Termination specifications for the velocity locomotion task."""

import math

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm
from mjlab.tasks.velocity import mdp

from mjlab_microduck.utils.configclass import configclass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: DoneTerm | None = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over: DoneTerm | None = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    out_of_terrain_bounds: DoneTerm | None = DoneTerm(
        func=mdp.out_of_terrain_bounds,
        time_out=True,
    )
