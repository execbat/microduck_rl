"""Command specifications for the Microduck ground_pick task.

Cyclic phase encoding (see ``microduck_flags.GP_PERIOD``/``DESCENT_END``/
``HOLD_END``/``RISE_END``): ``command = [cos(2*pi*phase), sin(2*pi*phase), 0]``,
phase in [0, 0.5) = approach (go down), [0.5, 1) = return (come back up).
All the base ``UniformVelocityCommandCfg`` fields not mentioned below
(``resampling_time_range``, ``rel_forward_envs``, ``heading_command``,
``heading_control_stiffness``, ``debug_vis``, ``ranges``) are inherited
unchanged from ``tasks/locomotion/velocity/cfg/commands_cfg.py`` -- the original file
never touched them, since ``GroundPickPhaseCommand.compute()`` overwrites
the whole command vector every step regardless.
"""

import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ground_pick.microduck_flags import GP_PERIOD
from mjlab_microduck.tasks.locomotion.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.GroundPickPhaseCommandCfg | None = microduck_mdp.GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),  # inherited unchanged from the base velocity command.
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,  # inherited unchanged.
        heading_command=True,  # inherited unchanged.
        heading_control_stiffness=0.5,  # inherited unchanged.
        debug_vis=True,
        ranges=microduck_mdp.GroundPickPhaseCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
        period=GP_PERIOD,
    )
