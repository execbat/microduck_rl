"""Termination specifications for the Microduck velocity task."""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.utils.configclass import configclass

from .cfg.terminations_cfg import TerminationsCfg


@configclass
class MicroduckTerminationsCfg(TerminationsCfg):
    """Termination terms for the Microduck velocity task."""

    # Terminate environments that have gone numerically unstable (NaN
    # physics). MuJoCo can produce NaN joint positions on extreme contact
    # impulses. Terminating immediately resets to a valid state before NaN
    # propagates into the observation buffer and corrupts network weights.
    nan_state: DoneTerm | None = DoneTerm(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )
