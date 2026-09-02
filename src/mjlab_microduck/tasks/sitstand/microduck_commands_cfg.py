"""Command specifications for the Microduck sitstand task.

``twist``: the sit/stand posture flag, ``cmd = [sit_flag, 0, 0]``; dwell-time
resampling flips the posture mid-episode. "Stand" is the all-zero command
(deployment idle parity). The runtime drives this by writing 0/1 into the
vx slot of the command buffer. Internally the term slews a target blend
over ``POSTURE_RAMP_S`` that the posture rewards track -- see
``SitStandCommand``'s docstring in ``mdp/commands.py``.

``head_pose``: commandable head control (like velocity/standup) -- new
field, not present on the base ``CommandsCfg`` (this is the first task with
two active commands).
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.sitstand.microduck_flags import POSTURE_DWELL_S, POSTURE_RAMP_S, SIT_PROB, SIT_Z, STAND_Z
from mjlab_microduck.tasks.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.tasks.velocity.microduck_flags import HEAD_POSE_CMD_RESAMPLE_S
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.SitStandCommandCfg | None = microduck_mdp.SitStandCommandCfg(
        entity_name="robot",
        resampling_time_range=POSTURE_DWELL_S,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,  # inherited unchanged from the base velocity command.
        heading_command=False,
        heading_control_stiffness=0.5,  # inherited unchanged.
        debug_vis=False,
        ranges=microduck_mdp.SitStandCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # inherited unchanged -- unused by SitStandCommand.
            lin_vel_y=(-1.0, 1.0),  # inherited unchanged -- unused by SitStandCommand.
            ang_vel_z=(-0.5, 0.5),  # inherited unchanged -- unused by SitStandCommand.
            heading=None,
        ),
        sit_prob=SIT_PROB,
        ramp_s=POSTURE_RAMP_S,
        sit_z=SIT_Z,
        stand_z=STAND_Z,
    )
    head_pose: microduck_mdp.UniformPoseCommandCfg | None = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )
