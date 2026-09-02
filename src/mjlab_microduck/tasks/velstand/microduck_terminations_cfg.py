"""Termination specifications for the Microduck velstand task.

Subclasses ``tasks.velocity``'s own ``MicroduckTerminationsCfg`` -- adds
``fallen_too_long`` (the failed-recovery backstop, Phase 2 of the module
docstring on ``microduck_velstand_env_cfg.py``). ``fell_over`` is inherited
unchanged here; the Flat/Rough env cfg classes handle disabling it via
curriculum (non-play) or dropping it outright (play) -- see
``microduck_velstand_env_cfg.py``.
"""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.microduck_terminations_cfg import MicroduckTerminationsCfg as _VelocityTerminationsCfg
from mjlab_microduck.tasks.velstand.microduck_flags import TERM_GATE_TILT_DEG, TERM_GATE_Z, FALLEN_TIMEOUT_S
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckTerminationsCfg(_VelocityTerminationsCfg):
    fallen_too_long: DoneTerm | None = DoneTerm(
        func=microduck_mdp.fallen_too_long,
        time_out=False,
        params={
            "gate_z_below": TERM_GATE_Z,
            "gate_tilt_above_deg": TERM_GATE_TILT_DEG,
            "max_duration_s": FALLEN_TIMEOUT_S,
        },
    )
