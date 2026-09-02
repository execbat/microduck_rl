"""Curriculum specifications for the velocity locomotion task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm
from mjlab.tasks.velocity import mdp

from mjlab_microduck.utils.configclass import configclass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Keep a disabled slot so enabling this in a subclass preserves term order.
    terrain_levels: CurrTerm | None = None
    command_vel: CurrTerm | None = CurrTerm(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
                {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
            ],
        },
    )


@configclass
class RoughCurriculumCfg(CurriculumCfg):
    """Common velocity curriculum plus procedural-terrain progression."""

    terrain_levels: CurrTerm | None = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"command_name": "twist"},
    )
