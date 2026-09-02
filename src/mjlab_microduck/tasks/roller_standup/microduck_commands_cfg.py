"""Command specifications for the Microduck roller_standup task.

The roller env installs a ``RelativeHeadingVelocityCommandCfg`` (cmd[2] =
internally-computed heading error). Here nothing is steered: switch back to
the neutralised command-only type, like ``standup``/``roulade``/``ball_kick``.
The head_pose (4) and body_pose (6) obs slots stay zero-padded -> 61D obs
parity preserved.

Fields not explicitly listed as changed below (``entity_name``,
``rel_forward_envs``, ``heading_control_stiffness``, ``viz``) are inherited
unchanged from ``velocity_rollers``'s command (the original code captured
them via ``vars(command)`` when converting the command TYPE).
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roller_standup.microduck_flags import EPISODE_LENGTH_S
from mjlab_microduck.tasks.locomotion.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.VelocityCommandCommandOnlyCfg | None = microduck_mdp.VelocityCommandCommandOnlyCfg(
        entity_name="robot",  # inherited unchanged.
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,  # inherited unchanged.
        heading_command=False,
        heading_control_stiffness=0.5,  # inherited unchanged.
        debug_vis=False,
        ranges=microduck_mdp.VelocityCommandCommandOnlyCfg.Ranges(
            lin_vel_x=(-0.01, 0.01),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.05, 0.05),
            heading=None,
        ),
        viz=microduck_mdp.VelocityCommandCommandOnlyCfg.VizCfg(z_offset=0.5),  # inherited unchanged.
    )
