"""Shared private helpers and constants used across the microduck MDP term
modules in this package.

Everything here is private (leading underscore, except the two shared
constants) and exists purely to avoid duplicating logic across
observations.py / rewards.py / events.py / terminations.py -- it is not part
of the public ``microduck_mdp.*`` surface."""

import math
import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_from_angle_axis

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Name patterns matching the 4 neck/head actuated joints. Used by head_pose
# tracking reward and by UniformPoseCommand asset hookups.
_NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """Entity-local indices of the servo (non-``passive_``) joints, cached.

    All joint-index-based reward/event params in this module (``joint_indices``,
    ``target_overrides``, qpos-column math) are written against the canonical
    14-servo layout. On models with extra unactuated joints — backlash hinges,
    roller wheels, the jaw linkage, all named ``passive_*`` — the entity joint
    array is wider and interleaved, so raw indices would select the wrong
    joints. Index through this list to recover the servo-only view; on plain
    models it is the identity.
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[key] = ids
    return ids



def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]



def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]



def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]



def _fallen_mask(
    env: ManagerBasedRlEnv,
    asset,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> torch.Tensor:
    """Per-env float mask: 1.0 where the robot counts as FALLEN — trunk height
    below `gate_z_below` OR tilt beyond `gate_tilt_above_deg`. Used to gate the
    recovery rewards so they only steer while actually fallen and contribute
    exactly zero during clean walking (no walk tax / bounce farming)."""
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()



def _multistage_target_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    waypoints,
) -> torch.Tensor:
    """Compute the time-interpolated joint target across N waypoints.

    waypoints: ordered list of dicts {"frac": float in [0,1],
                                       "overrides": dict[int,float] | None}.
    First waypoint should have frac=0.0 (typically HOME, overrides=None).
    Subsequent waypoints define milestones. Between two waypoints the target
    linearly interpolates. Before the first / after the last it clamps.

    Returns a (num_envs, num_joints) tensor of target joint angles.
    """
    asset = env.scene[asset_cfg.name]
    default = _servo_default_joint_pos(env, asset)

    def build_pose(overrides):
        pose = default.clone()
        if overrides:
            for idx, val in overrides.items():
                pose[:, idx] = val
        return pose

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    # Find which segment we're in (broadcast over envs).
    out = build_pose(waypoints[0]["overrides"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0).unsqueeze(-1)
        prev_pose = build_pose(waypoints[i - 1]["overrides"])
        next_pose = build_pose(waypoints[i]["overrides"])
        seg = prev_pose * (1.0 - tau) + next_pose * tau
        # Take this segment's value when progress is in [f0, f1] or past it.
        mask = (progress >= f0).float().unsqueeze(-1)
        out = torch.where(mask > 0, seg, out)
    return out



def _multistage_target_height(
    env: ManagerBasedRlEnv,
    waypoints,
) -> torch.Tensor:
    """Same logic as _multistage_target_pose but for trunk z height.

    waypoints: [{"frac": float, "height": float}, ...].
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = torch.full_like(progress, waypoints[0]["height"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0)
        seg = waypoints[i - 1]["height"] * (1.0 - tau) + waypoints[i]["height"] * tau
        mask = (progress >= f0).float()
        out = torch.where(mask > 0, seg, out)
    return out



def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0



def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    """Per-env constant IMU mounting-misalignment rotation (sampled once).

    Models a fixed small mounting/calibration error of the IMU on each robot.
    Sampled lazily on first use and cached — constant per env for the whole run
    (like a startup randomization), so it's a *systematic per-robot bias*, not
    per-step noise. Replaces the old randomize_imu_orientation event, which wrote
    site_quat (not per-env expanded under mjlab 1.3.0, and not read by the
    projected_gravity / base_ang_vel observations anyway).

    Returns a (num_envs, 4) unit quaternion (w, x, y, z).
    """
    q = getattr(env, "_imu_misalign_quat", None)
    if q is None:
        n = env.num_envs
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(n, device=env.device) * max_angle_rad  # [0, max]
        q = quat_from_angle_axis(angle, axis)
        env._imu_misalign_quat = q
    return q



def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    """0→1 ramp in body forward speed: 0 when standing still, 1 at/above v_ref.

    Used to gate stride-shaping rewards so that stepping which does NOT propel
    the body (e.g. tap-dancing on the spot) earns nothing — the reward for the
    FORM of a stride is only paid when the stride actually does its JOB (moving
    forward). Returns None when disabled (v_ref <= 0)."""
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)



def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)



def _ball_kick_dir(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-env world-frame kick direction (XY unit vector), lazily allocated.

    Set by ``reset_ball_in_front_of_foot`` to the robot's forward direction at
    episode reset. Frozen for the episode so the policy can't redefine "forward"
    by turning after the kick.
    """
    if not hasattr(env, "_ball_kick_dir_w"):
        env._ball_kick_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir_w[:, 0] = 1.0
    return env._ball_kick_dir_w



def _backlash_encoder_ids(
    env: "ManagerBasedRlEnv",
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(main_ids, backlash_ids, mask) — cached per (entity, joint selection).

    mask is 1.0 where a matching passive_<name>_backlash joint exists, so the
    same obs functions run unchanged on models without backlash joints.
    """
    key = (asset_cfg.name, str(asset_cfg.joint_ids))
    cache = env.__dict__.setdefault("_backlash_encoder_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit

    names = asset.joint_names
    jnt_ids = asset_cfg.joint_ids
    if isinstance(jnt_ids, slice):
        main_ids = list(range(len(names)))[jnt_ids]
    else:
        main_ids = [int(i) for i in jnt_ids]
    name_to_id = {n: i for i, n in enumerate(names)}
    bl_ids, mask = [], []
    for i in main_ids:
        bl = name_to_id.get(f"passive_{names[i]}_backlash")
        bl_ids.append(0 if bl is None else bl)
        mask.append(0.0 if bl is None else 1.0)

    device = asset.data.joint_pos.device
    out = (
        torch.tensor(main_ids, dtype=torch.long, device=device),
        torch.tensor(bl_ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )
    cache[key] = out
    return out



def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target blend ∈ [0, 1] (0 = STAND, 1 = SIT) for the posture rewards.

    Uses the SitStandCommand's slewed ``alpha`` (the moving setpoint) when the
    term exposes it; falls back to the raw binary flag otherwise.
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]



def _posture_targets(
    env: ManagerBasedRlEnv,
    asset: Entity,
    command_name: str,
    sit_overrides: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(target blend, per-env joint target) for the commanded posture.

    STAND target = default_joint_pos (HOME); SIT target = HOME with the
    keyframe overrides applied; the SLEWED blend interpolates between them,
    so mid-ramp the rewarded pose folds in sync with the descending height.
    """
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target



def _posture_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(slewed target trunk z, actual trunk z) per env."""
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return target_z, z



# ==============================================================================
# Roulade (forward roll) task — episodic dynamic maneuver
# ==============================================================================
#
# Third attempt at the roulade. What the first two taught us:
#   • origin/roulade (phase-clock + time-windowed reward stages): plateaued
#     face-down at ~90° — time windows are keyframes-in-time, campable local
#     optima (the sit/standup lesson exactly). Also integrated -ω_y as forward
#     progress, which by this codebase's own convention (face-down = +90° pitch
#     = rotation about +y, see set_random_ground_state) is the WRONG SIGN — the
#     progress reward paid for backward rotation.
#   • origin/roulade later commits (keyframe imitation): same waypoint-camping
#     family, dropped per feedback-episodic-pose-landing.
#
# This design uses the proven episodic recipe instead:
#   • ONE dense progress signal: paid INCREMENTS of the max-so-far cumulative
#     forward rotation (potential-based — a camping policy earns zero/step, a
#     full roll earns exactly 2π worth no matter the path or speed).
#   • Landing rewards (composite product, upright, height, rise velocity) are
#     gated on ROLL COMPLETION (max rotation ≥ threshold) — state-based gates,
#     not clock-based. "Do nothing" earns nothing; standing at spawn earns
#     nothing; only rolling opens the standing-attractor annuity.
#   • Reverse curriculum via mid-roll spawns (the face-up partial-roll trick
#     that fixed back-recovery): a slice of episodes starts pitched 50°–185°
#     into the roll, tucked, optionally with forward angular momentum, and the
#     rotation accumulator is initialized to the spawn angle so the progress
#     accounting stays consistent.
#
# RUN-1 LESSON (2026-08): with unsupported rotation counting and uncapped
# paid rate, the optimal policy is a violent ballistic whip ("breakdance") —
# same 2π, finishes sooner, more discounted annuity. Doesn't transfer. Fixes:
#   • SUPPORT GATE: the accumulator only integrates while some robot geom
#     touches the terrain (robot_ground_contact sensor) — a real roulade never
#     leaves the ground; airborne rotation now earns nothing and cannot open
#     the completion gate.
#   • HEAD LATCH: the landing annuity additionally requires head-ground
#     contact to have occurred while accum was in the first-quadrant window —
#     "went over the head" is a requirement, not a 0.5-weight suggestion.
#   • PAID-RATE CAP: progress increments are capped at max_paid_rate; rotation
#     faster than the cap FORFEITS the excess (not deferred), so speed no
#     longer pays. An explicit overspeed penalty backs this up.
#
# Per-env state on the env object (created lazily, reset by
# reset_roulade_state):
#   env._roulade_accum      — supported-only integral of forward pitch rate (rad)
#   env._roulade_max        — max(accum) so far this episode (progress frontier)
#   env._roulade_paid       — frontier already paid out by roulade_progress
#   env._roulade_head_latch — True once the head touched ground mid-first-quadrant

# Forward-roll sign: face-down is +90° pitch = rotation about body +y
# (set_random_ground_state convention), so forward roll = POSITIVE body-frame
# ω_y. Verified empirically (see claude_experiments smoke test): a positive
# qvel about +y pitches the robot nose-down/forward and drives accum upward.
_ROULADE_FWD_SIGN = 1.0

# Sensor names read by the accumulator update (must match the env cfg).
_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"
_ROULADE_HEAD_SENSOR = "head_ground_contact"

# Head-latch window: head-ground contact while accum is inside this window
# marks the episode as a genuine over-the-head roll. In a real roulade the
# head plants at ~60–120° of body rotation; the window is generous around it.
_HEAD_LATCH_LO = math.radians(20.0)
_HEAD_LATCH_HI = math.radians(170.0)

# Head-top axis in jaw_soft's LOCAL frame (measured empirically 2026-08-13:
# world-up expressed in jaw_soft's frame with the robot settled at HOME).
# The latch requires this axis to point DOWN at contact — "the flat top of
# the head on the floor", not the face or the side of the shell (run-5 fix:
# the run-4 policy rolled over the shoulder, which still touched jaw_soft).
_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
# dot(top_axis_world, -z) threshold. Measured landmarks (trunk pitched 110°):
# passive face-plant (neck at HOME) reads +0.6, full chin-tuck (neck_pitch −1,
# head_pitch +1) reads −0.99 — 0.3 accepts partial tucks while staying far
# from any face/side contact.
_HEAD_TOP_DOWN_MIN = 0.3

# Sagittal flatness gate on the accumulator (run-5): in a clean forward roll
# the body's LATERAL axis stays horizontal the whole way — its world-z
# component is 2(q_y·q_z + q_w·q_x) ≈ 0 for ANY amount of pure pitch, and
# grows toward ±1 as the roll goes over the shoulder instead. Full rotation
# credit while the lateral axis is within ~30° of horizontal, zero beyond
# ~60°: a side roll does not count as rotation, earns no progress, and never
# opens the landing gate.
_FLAT_FULL = 0.5    # |lateral_axis_z| = sin(30°): full credit below
_FLAT_ZERO = 0.866  # sin(60°): zero credit above



def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z component of the body's lateral (y) axis. 0 = flat/sagittal."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])



def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """True where the head-top axis points at the floor (dot with -z > min)."""
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
    q = asset.data.body_link_quat_w[:, env._roulade_head_body_id]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a, b, c = _HEAD_TOP_AXIS
    # z-component of R(q) @ axis_local
    axis_world_z = (
        2.0 * (x * z - w * y) * a + 2.0 * (y * z + w * x) * b + (1.0 - 2.0 * (x * x + y * y)) * c
    )
    return axis_world_z < -_HEAD_TOP_DOWN_MIN



def _sensor_any_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
    if name not in env.scene.sensors:
        return None
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)



def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    return env._roulade_accum, env._roulade_max, env._roulade_paid



def _update_roulade_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Integrate forward pitch rate into the per-env rotation accumulator.

    Step-guarded so that multiple reward terms reading the accumulator in the
    same control step don't double-integrate. The frontier (max) only moves
    forward; backward rocking (wind-up) neither pays nor un-pays.

    SUPPORT GATE (run-1 fix): rotation is integrated only while the robot
    touches the terrain — a roulade is a supported motion; ballistic flips
    accumulate nothing, so they neither get paid nor open the completion gate.

    Also latches env._roulade_head_latch when the head touches the ground
    while accum is inside the first-quadrant window — the landing annuity
    requires this, making "over the head" a hard requirement of the task.
    """
    _roulade_state(env)
    step = int(env.common_step_counter)
    if step != env._roulade_last_update_step:
        omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
        delta = torch.nan_to_num(omega_fwd, nan=0.0) * env.step_dt
        supported = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
        if supported is not None:
            delta = delta * supported.float()
        # Sagittal flatness gate (run-5): side/shoulder rolls don't count.
        y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
        t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
        delta = delta * (t * t * (3.0 - 2.0 * t))
        env._roulade_accum = env._roulade_accum + delta
        env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)

        head_contact = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
        if head_contact is not None:
            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (
                env._roulade_accum < _HEAD_LATCH_HI
            )
            # Run-5: contact must be with the FLAT TOP of the head (top axis
            # pointing at the floor) — face/side shell contacts don't latch.
            env._roulade_head_latch = env._roulade_head_latch | (
                head_contact & in_window & _head_top_down(env, asset)
            )
        env._roulade_last_update_step = step



def _roulade_completion_gate(
    env: ManagerBasedRlEnv,
    gate_lo: float,
    gate_hi: float,
    require_head: bool = False,
) -> torch.Tensor:
    """Smoothstep on the progress frontier: 0 below gate_lo rad, 1 above gate_hi.

    State-based replacement for the old phase-clock landing window — it can
    only be opened by actually rotating (while SUPPORTED — the accumulator is
    contact-gated), so neither pre-roll standing nor a ballistic flip collects.
    With require_head=True the gate additionally requires the head latch —
    the episode must have rolled over the head to unlock the landing annuity.
    """
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate
