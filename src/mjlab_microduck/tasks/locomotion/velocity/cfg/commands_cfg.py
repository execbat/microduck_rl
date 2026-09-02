"""Command specifications for the velocity locomotion task."""

import math

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.utils.configclass import configclass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    twist: UniformVelocityCommandCfg | None = UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.3,
        rel_forward_envs=0.2,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
