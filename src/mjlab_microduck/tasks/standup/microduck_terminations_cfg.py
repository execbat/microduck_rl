"""Termination specifications for the Microduck standup task.

Robot starts seated (or prone, for recovery training) -- tilt-based fall
termination doesn't apply here, so ``fell_over`` is dropped.
"""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.cfg.terminations_cfg import TerminationsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckTerminationsCfg(TerminationsCfg):
    fell_over: DoneTerm | None = None
    nan_state: DoneTerm | None = DoneTerm(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )
