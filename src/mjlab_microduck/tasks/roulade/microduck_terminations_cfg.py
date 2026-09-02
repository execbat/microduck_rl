"""Termination specifications for the Microduck roulade task.

Falling over is the task -- ``fell_over`` is dropped entirely. Only the NaN
guard and the base timeout remain (``time_out``/``out_of_terrain_bounds``
inherited unchanged from the base ``TerminationsCfg``).
"""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.cfg.terminations_cfg import TerminationsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckTerminationsCfg(TerminationsCfg):
    fell_over: DoneTerm | None = None
    nan_state: DoneTerm | None = DoneTerm(func=microduck_mdp.robot_state_is_nan, time_out=False)
