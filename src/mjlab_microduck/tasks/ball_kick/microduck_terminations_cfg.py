"""Termination specifications for the Microduck BallKick task.

``fell_over`` is kept inherited unchanged from the base velocity
``TerminationsCfg`` -- the robot starts standing and must stay up through the
kick. ``out_of_terrain_bounds`` is also left inherited unchanged (the
original env never touched it, even on flat terrain).
"""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.utils.configclass import configclass

from mjlab_microduck.tasks.velocity.cfg.terminations_cfg import TerminationsCfg


@configclass
class MicroduckTerminationsCfg(TerminationsCfg):
    nan_state: DoneTerm | None = DoneTerm(func=microduck_mdp.robot_state_is_nan, time_out=False)
