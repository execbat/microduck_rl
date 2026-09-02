"""Tunables for the Microduck roller_standup task.

Direct port of the module-level constants from the old
``tasks/microduck_roller_standup_env_cfg.py``. Dedicated episodic policy:
the robot starts on the ground (face-down, face-up, or already standing)
and must get back up onto its rollers and then HOLD the stance. Port of the
walking-duck ``standup`` recipe to the roller model.

Builds on the roller env (``tasks.velocity_rollers``) -- inherits the
roller robot, sensors, all DR, and the unified 61D observation as-is, so
it's hot-swappable at runtime (--new-cmd-obs). Same pattern as
``roller_slope``.

Two structural differences from ``standup``:
  - the passive wheels are INTERLEAVED in joint order -> remapped indices
    (``LEG_JOINTS`` below), locked by ``tests/test_roller_standup_cfg.py``;
  - no ``head_pose`` command: the head/body slots stay zero-padded (the
    roller family convention) and the head is held straight by
    ``neck_joint_pos_l2``, which resolves by NAME.

The genuinely new piece is the wheel-friction curriculum, INVERTED (braked
-> free): the wheels roll, so there's zero grip to push off the ground
with. Bootstrap with near-locked wheels, then ramp toward the real value.
If ``standing_composite`` collapses at a stage, the "sticky feet" gesture
doesn't transfer and a skater technique (knee support, one skate at a time)
will need to be guided in.

Deployment target: in ``--standing`` facing the roller policy in
``--walking``, with automatic hand-off on velocity-command magnitude
(infer_policy.py:262, threshold 0.05); the twist slot is left at zero there
(infer_policy.py:239).
"""

import os

# ── Trunk heights (m) ────────────────────────────────────────────────────────
# Measured by exact kinematics (minimum of colliding-geom mesh vertices,
# STAND pose, trunk brought into contact) on scene_rollers.xml: standing
# 0.1407, resting face-down 0.0752, resting face-up 0.0475.
# Sanity check: the model WITHOUT wheels gives 0.1172 kinematically vs. the
# STAND_Z=0.115 measured under load by standup -> ~2mm sag, applied here too.
# 0.138 falls inside the reset_base z range (0.1335-0.1435) already used by
# the roller env.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S = 6.0  # rise + stabilise, like standup
NUM_STEPS_PER_ENV = 24

# ── Play override: force the proportion of FACE-UP (on-the-back) starts ------
# In play mode the env is rebuilt from scratch: common_step_counter restarts
# at 0, so the ground_state_mix curriculum applies its stage 0, where
# face_up_prob = 0. So you NEVER see a back-start in play -- yet that's the
# hardest case, the one you want to eyeball. This variable forces it.
#   STANDUP_PLAY_FACE_UP=1.0  -> 100% back starts
#   STANDUP_PLAY_FACE_UP=0.4  -> the mix of the curriculum's last stage
#   unset / "none" / "random" -> default behavior (stage 0)
# Only affects play=True. Same pattern as SLOPE_PLAY_DIFFICULTY in roller_slope.
PLAY_FACE_UP = None
# belly:standing ratio of the curriculum's LAST stage (0.40 / 0.20 = 2:1).
# The remainder (1 - face_up) is split in this ratio, so 0.4 exactly
# reproduces the end-of-training mix.
_PLAY_FACE_DOWN_SHARE = 2.0 / 3.0


def resolve_play_face_up():
    """Proportion of back-starts in play mode: STANDUP_PLAY_FACE_UP env var, else the constant."""
    raw = os.environ.get("STANDUP_PLAY_FACE_UP")
    if raw is None:
        return PLAY_FACE_UP
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_standup] STANDUP_PLAY_FACE_UP='{raw}' invalid -- falling back to {PLAY_FACE_UP}")
        return PLAY_FACE_UP


def play_face_up_overrides(play_face_up: float) -> dict:
    """(face_up_prob, face_down_prob, standing_prob) split for a forced play override."""
    remainder = 1.0 - play_face_up
    return {
        "face_up_prob": play_face_up,
        "face_down_prob": remainder * _PLAY_FACE_DOWN_SHARE,
        "standing_prob": remainder * (1.0 - _PLAY_FACE_DOWN_SHARE),
        "sitting_prob": 0.00,
    }


# ── Joint indices -- the passive wheels are INTERLEAVED ----------------------
# Real joint order of the roller model (18 joints after the free joint),
# verified in MuJoCo via get_walk_rollers_spec().compile():
#   0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
#   5-6   passive_LF_wheel, passive_LR_wheel
#   7-10  neck_pitch, head_pitch, head_yaw, head_roll
#   11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
#   16-17 passive_RF_wheel, passive_RR_wheel
# standup uses [0-4, 9-13] / [5-8]: those are the WHEEL-LESS model's indices,
# they do NOT apply here. Locked by tests/test_roller_standup_cfg.py.
#
# Only LEG_JOINTS is actually consumed (by the pose rewards). NECK_JOINTS and
# WHEEL_JOINTS exist for documentation and the index-lock test: the neck is
# resolved by NAME (neck_joint_pos_l2 calls find_joints(r".*(neck|head).*")
# every step) and the wheels by the ^passive_.* regex.
LEG_JOINTS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
NECK_JOINTS = [7, 8, 9, 10]
WHEEL_JOINTS = [5, 6, 16, 17]

# -- Servo-local (compacted) indices, for reward params -----------------------
# BUGFIX (not present in the original file -- caught by a real CUDA
# "index out of bounds" crash on pose_target_match during an actual GPU
# training run, post-migration): pose_target_match/pose_l1_penalty/
# standing_composite_score all route through _servo_joint_pos()/
# _servo_default_joint_pos() (see mdp/_common.py), which ALREADY filters out
# the passive_* wheel joints and compacts the remaining 14 servo joints down
# to a dense 0-13 index space -- "on models with extra unactuated joints...
# raw indices would select the wrong joints. Index through this list to
# recover the servo-only view" (that function's own docstring). LEG_JOINTS
# above is the RAW/full 18-joint layout (correct for the joint-index-lock
# test, which compiles the real MjSpec and checks against ITS index space)
# -- but it is NOT what these three reward functions expect, and index 14/15
# don't even exist in a 14-column servo-filtered tensor. This is what SHOULD
# be passed as their `joint_indices` param: the servo-local equivalent,
# which -- because filtering out the interleaved wheels leaves the same
# leg/neck/leg joint order as the wheel-less model -- is numerically
# IDENTICAL to `standup`'s own LEG_JOINTS.
SERVO_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Roller-env SKATING rewards: meaningless while on the ground.
# feet_flat: the blades are NOT flat during the rise -> would fight the motion.
# hip_roll_neutral: getting up requires splitting the legs apart.
# pose / com_height_target: replaced by the rise's own pose/height targets.
# upright (base gaussian): replaced by upright_linear + upright_sharp.
SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)
