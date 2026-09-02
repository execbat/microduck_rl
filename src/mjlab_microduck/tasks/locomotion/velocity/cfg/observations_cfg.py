"""Observation specifications for the velocity locomotion task.

Common defaults omit body terrain scans; rough variants opt into them. A
robot-specific env cfg (e.g. ``MicroduckObservationsCfg``) subclasses
``PolicyCfg``/``CriticCfg`` to add/remove/override terms.
"""

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.utils.configclass import configclass
from mjlab_microduck.utils.manager_compat import ObsGroup

# height_scan's noise/scale reference the terrain raycast sensor's max_distance.
# Kept as a plain constant here (rather than importing the sensor object) so
# this file has no dependency on scene wiring.
_TERRAIN_SCAN_MAX_DISTANCE = 5.0


@configclass
class PolicyCfg(ObsGroup):
    """Observations for the policy ("actor") group."""

    base_lin_vel: ObsTerm | None = ObsTerm(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    base_ang_vel: ObsTerm | None = ObsTerm(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm | None = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm | None = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions: ObsTerm | None = ObsTerm(func=mdp.last_action)
    command: ObsTerm | None = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
    )
    # Reserve the field position to preserve observation order in subclasses.
    height_scan: ObsTerm | None = None

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticCfg(PolicyCfg):
    """Observations for the ("critic") value-function group.

    Same terms as the policy group by default, plus privileged extras (no
    noise/corruption -- the critic sees ground truth).
    """

    foot_height: ObsTerm | None = ObsTerm(
        func=mdp.foot_height,
        params={"sensor_name": "foot_height_scan"},
    )
    foot_air_time: ObsTerm | None = ObsTerm(
        func=mdp.foot_air_time,
        params={"sensor_name": "feet_ground_contact"},
    )
    foot_contact: ObsTerm | None = ObsTerm(
        func=mdp.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    )
    foot_contact_forces: ObsTerm | None = ObsTerm(
        func=mdp.foot_contact_forces,
        params={"sensor_name": "feet_ground_contact"},
    )

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    actor: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RoughPolicyCfg(PolicyCfg):
    """Actor observations including a noisy body terrain scan."""

    height_scan: ObsTerm | None = ObsTerm(
        func=envs_mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        scale=1 / _TERRAIN_SCAN_MAX_DISTANCE,
    )


@configclass
class RoughCriticCfg(CriticCfg):
    """Privileged observations including an uncorrupted body terrain scan."""

    height_scan: ObsTerm | None = ObsTerm(
        func=envs_mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        scale=1 / _TERRAIN_SCAN_MAX_DISTANCE,
    )


@configclass
class RoughObservationsCfg(ObservationsCfg):
    """Observation groups for the generic rough-terrain template."""

    actor: RoughPolicyCfg = RoughPolicyCfg()
    critic: RoughCriticCfg = RoughCriticCfg()
