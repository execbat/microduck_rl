"""Observation terms for the microduck MDP (``ObservationTermCfg.func``)."""

import math
import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp import observations as _velocity_obs
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply

from mjlab_microduck.tasks.mdp._common import (
    _DEFAULT_ASSET_CFG,
    _backlash_encoder_ids,
    _finite,
    _imu_misalignment_quat,
)


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Projected gravity vector in body frame.

    Returns the gravity vector projected into the robot's body frame,
    representing pure orientation without linear acceleration.
    This is simpler than raw accelerometer and only depends on orientation.

    Returns:
        torch.Tensor: Projected gravity in body frame (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b



def projected_gravity_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """projected_gravity with a per-env constant IMU mounting misalignment."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.projected_gravity_b)



def base_ang_vel_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """base angular velocity with the SAME per-env IMU misalignment as gravity."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.root_link_ang_vel_b)



def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw accelerometer reading (includes gravity + linear acceleration).

    Returns normalized raw accelerometer which mimics what a real IMU measures.
    This is different from pure projected_gravity which only reflects orientation.
    Reads from the MuJoCo accelerometer sensor "imu_accel".

    Returns:
        torch.Tensor: Normalized raw accelerometer reading (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Access the model to find the sensor address
    # The accelerometer sensor is the 5th sensor (index 4) in robot.xml
    # Sensors: framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    # Get sensor address from model arrays (sensor_adr is torch tensor)
    sensor_adr_array = mj_model.sensor_adr  # This is a TorchArray/tensor
    sensor_id = 4  # imu_accel is the 5th sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())  # Convert to Python int

    # Read accelerometer data (specific force measured by sensor)
    # Shape: (num_envs, 3)
    accel_raw = asset.data.data.sensordata[:, sensor_adr:sensor_adr+3]

    # MuJoCo accelerometer measures specific force (like real sensor)
    # Negate to match convention: when at rest upright, should point down
    accel_negated = -accel_raw

    # Normalize to unit vector
    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b  # Fallback to projected gravity
    )

    return accel_normalized


def zero_command_padding(
    env: ManagerBasedRlEnv,
    dim: int,
) -> torch.Tensor:
    """Constant-zero obs term of width `dim`.

    Used by envs that don't actively track head/body commands (e.g. sitstand,
    ground_pick) but still need the unified 61D obs shape so the runtime can
    feed all policies with the same buffer layout.
    """
    return torch.zeros(env.num_envs, dim, device=env.device)



def foot_contact_forces_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_contact_forces` (see note above)."""
    return _finite(_velocity_obs.foot_contact_forces(env, sensor_name))



def foot_height_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_height` (see note above)."""
    return _finite(_velocity_obs.foot_height(env, sensor_name))



def foot_air_time_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_air_time` (see note above)."""
    return _finite(_velocity_obs.foot_air_time(env, sensor_name))



def ball_pos_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """Ball position relative to the robot root, in the robot's base frame.

    CRITIC-ONLY observation (asymmetric actor-critic): the deployed policy has
    no ball sensing, so the actor must stay blind to the ball — the critic can
    still use it to predict the kick payoff.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)



def ball_vel_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """Ball linear velocity in the robot's base frame. CRITIC-ONLY (see above)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    vel = ball.data.root_link_lin_vel_w
    return torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)

def joint_pos_rel_backlash(
    env: "ManagerBasedRlEnv",
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """joint_pos_rel where the encoder reads through the backlash hinge.

    Returns (qpos[servo] + qpos[backlash]) - default[servo]. With biased=True
    the per-env encoder-calibration bias is applied to the servo reading (one
    encoder per servo → one bias per joint; the backlash summand stays raw).
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    pos = joint_pos[:, main_ids] + asset.data.joint_pos[:, bl_ids] * mask
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return pos - default_joint_pos[:, main_ids]



def joint_vel_rel_backlash(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """joint_vel_rel where the encoder reads through the backlash hinge.

    The firmware derives present_velocity from encoder positions, so it also
    sees the backlash motion: qvel[servo] + qvel[backlash].
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    vel = asset.data.joint_vel[:, main_ids] + asset.data.joint_vel[:, bl_ids] * mask
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return vel - default_joint_vel[:, main_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Sit↔Stand posture command + posture-conditioned rewards (sitstand env).
#
# One policy, both directions: the command is a single sit/stand flag carried
# in the twist slot (cmd = [sit_flag, 0, 0], so "stand" is the all-zero
# command — same deployment idle as every other policy). All task rewards
# below select their target (SIT keyframe + SIT_Z vs HOME + STAND_Z) from the
# live command, per env, so the same reward stack drives the descent, the
# seated rest, the rise and the standing rest. Uses the _servo_* helpers →
# backlash-model compatible.
# ─────────────────────────────────────────────────────────────────────────────
