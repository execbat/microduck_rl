"""Curriculum specifications for the Microduck velstand task.

Subclasses ``tasks.velocity``'s own ``MicroduckCurriculumCfg`` -- adds the
recovery-specific curricula. ``fell_over_disable`` is declared here (ON by
default); the play-mode env cfg variants set it to ``None`` in their own
``__post_init__`` instead (the curriculum doesn't run in play mode, so the
disable would never fire -- see ``microduck_velstand_env_cfg.py``).
"""

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.microduck_curriculum_cfg import MicroduckCurriculumCfg as _VelocityCurriculumCfg
from mjlab_microduck.tasks.velstand.microduck_flags import (
    FELL_OVER_DISABLE_ITER,
    NUM_STEPS_PER_ENV,
    PRONE_RAMP_STAGES,
    RECOVERY_ECON_KICKIN_ITER,
)
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCurriculumCfg(_VelocityCurriculumCfg):
    # Phase 1 -> 2: disable fell_over at iter 500 (limit 70deg -> 180deg) so
    # falls become recovery training instead of episode ends.
    fell_over_disable: CurrTerm | None = CurrTerm(
        func=microduck_mdp.termination_param_curriculum,
        params={
            "term_name": "fell_over",
            "param_stages": [
                {"step": 0, "params": {"limit_angle": math.radians(70.0)}},
                {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV, "params": {"limit_angle": math.pi}},
            ],
        },
    )
    # Phase 3: prone-init ramp (face-down first, face-up later, capped 45%).
    prone_init_prob: CurrTerm | None = CurrTerm(
        func=microduck_mdp.event_param_curriculum,
        params={"event_name": "random_prone_init", "param_stages": PRONE_RAMP_STAGES},
    )
    # Recovery economics ramp: tax + bounty OFF until the walk is
    # established (run-3 crouch-freeze lesson -- see microduck_flags.py's
    # RECOVERY_ECON_KICKIN_ITER note).
    fallen_tax_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "fallen_tax",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )
    recovery_success_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "recovery_success",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0},
            ],
        },
    )
    com_upward_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "com_upward_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0},
            ],
        },
    )
