"""Command specifications for the Microduck velocity_swizzle task.

``twist``: all fields inherited unchanged from ``velocity_rollers``'s
``RelativeHeadingVelocityCommandCfg`` except ``ranges.lin_vel_x`` (-0.5,0.6
-> -0.6,0.6, symmetrised so forward and backward get equal push range --
see the backward-locomotion note in ``microduck_rewards_cfg.py``'s
``wheel_speed`` override) and ``ranges.ang_vel_z`` (0,0 -> -0.5,0.5,
re-enabling turning -- see the heading curriculum in
``microduck_curriculum_cfg.py``).

``head_pose``: new field (velocity_rollers has no head command) -- real
commandable head control, ramped in by curriculum once the swizzle gait is
solid (see ``head_pose_range``/``head_pose_tracking_weight`` in
``microduck_curriculum_cfg.py``).
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.locomotion.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.tasks.velocity_swizzle.microduck_flags import HEAD_POSE_INITIAL_RANGES
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.RelativeHeadingVelocityCommandCfg | None = (
        microduck_mdp.RelativeHeadingVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),  # inherited unchanged.
            rel_standing_envs=0.0,  # inherited unchanged.
            rel_heading_envs=0.0,  # inherited unchanged.
            rel_forward_envs=0.2,  # inherited unchanged.
            heading_command=False,  # inherited unchanged.
            heading_control_stiffness=0.5,  # inherited unchanged.
            debug_vis=True,  # inherited unchanged.
            ranges=microduck_mdp.RelativeHeadingVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.6, 0.6),
                lin_vel_y=(0.0, 0.0),  # inherited unchanged.
                ang_vel_z=(-0.5, 0.5),
                heading=None,  # inherited unchanged.
            ),
            viz=microduck_mdp.RelativeHeadingVelocityCommandCfg.VizCfg(z_offset=0.5),  # inherited unchanged.
        )
    )
    head_pose: microduck_mdp.UniformPoseCommandCfg | None = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=HEAD_POSE_INITIAL_RANGES,
    )
