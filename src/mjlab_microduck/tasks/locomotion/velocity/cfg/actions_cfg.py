"""Action specifications for the velocity locomotion task."""

from mjlab.envs.mdp.actions import JointPositionActionCfg

from mjlab_microduck.utils.configclass import configclass


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    joint_pos: JointPositionActionCfg | None = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.5,  # Override per-robot.
        use_default_offset=True,
    )
