"""Termination specifications for the Microduck velocity_rollers task."""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.locomotion.velocity.cfg.terminations_cfg import TerminationsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckTerminationsCfg(TerminationsCfg):
    nan_state: DoneTerm | None = DoneTerm(func=microduck_mdp.robot_state_is_nan, time_out=False)
