"""Tunables for the Microduck roller_crouch (crouch-glide) task.

One-shot trick triggered by button A via the runtime's --ground-pick slot:
the robot crouches and glides on its momentum (a ~1s hold), then rises and
hands control back to the roller policy.

Hybrid task, built from two other tasks' machinery:
  - physics / roller robot          <- velocity_rollers
  - one-shot phase machinery         <- ground_pick
    (GroundPickPhaseCommand: [cos(2*pi*phase), sin(2*pi*phase), 0], see below)

Trapezoid height target (up -> down -> ~1s hold -> up) via
``crouch_glide_pose_by_phase``. Unified 61D obs -> hot-swappable at runtime.
"""

ENABLE_SYMMETRY = False

# -- Domain randomisation (matched to velocity_rollers) ----------------------
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_WHEEL_FRICTION_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.2, 0.2)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)

# m/s: the robot arrives already rolling.
ENTRY_VELOCITY_X = (0.2, 0.5)

# Phase timing, 4 segments over a 5s period:
#   descent [0, DESCENT_END]        = 0.10*5 = 0.5s  (crouch down)
#   low/hold [DESCENT_END, HOLD_END] = 0.40*5 = 2.0s  (crouched glide)
#   rise    [HOLD_END, RISE_END]    = 0.10*5 = 0.5s  (stand back up)
#   high/stand [RISE_END, 1.0]      = 0.40*5 = 2.0s  (standing rest)
# NOTE: this period MUST match --ground-pick-period at deployment (5.0).
CROUCH_PERIOD = 5.0
DESCENT_END = 0.10
HOLD_END = 0.50
RISE_END = 0.60

# Target CROUCH pose (rad, by joint NAME -- composed in
# scripts/crouch_pose_editor.py). The reward interpolates STAND(HOME) <->
# this pose by phase. Name-based resolution -> robust to interleaved wheels.
# STAND pose (trick start/end). Default = sim HOME (validated convention,
# equal to the real-robot reading). Replace with a read_pose.py capture of
# the real standing robot if a different stand is wanted.
# NOTE: at deployment, at the end of the trick the runtime hands control
# back to the roller policy, which starts from HOME -- keep STAND_POSE close
# to HOME for a clean handoff.
STAND_POSE = {
    # Read from the REAL robot (read_pose.py) -- the desired stand for the trick.
    "left_hip_yaw": -0.0476, "left_hip_roll": -0.0629, "left_hip_pitch": -0.2869,
    "left_knee": 0.9618, "left_ankle": 1.1674,
    "neck_pitch": 0.6029, "head_pitch": 0.543, "head_yaw": -0.069, "head_roll": -0.0414,
    "right_hip_yaw": -0.0337, "right_hip_roll": -0.0061, "right_hip_pitch": 0.1534,
    "right_knee": -0.9725, "right_ankle": -1.0646,
}

CROUCH_POSE = {
    # Read from the REAL robot (Dynamixel XL330, read_pose.py) -- a holdable pose.
    "left_hip_yaw": -0.0184,
    "left_hip_roll": 0.0307,
    "left_hip_pitch": 1.4082,
    "left_knee": 1.5248,
    "left_ankle": -0.0675,
    "neck_pitch": 1.0937,
    "head_pitch": 1.2149,
    "head_yaw": -0.0184,
    "head_roll": -0.0368,
    "right_hip_yaw": 0.0184,
    "right_hip_roll": -0.0169,
    "right_hip_pitch": -1.4757,
    "right_knee": -1.5907,
    "right_ankle": 0.0568,
}
CROUCH_POSE_STD = 0.4  # per-joint gaussian tolerance (rad)
CROUCH_LEAN_PITCH = 0.08  # slight forward lean while crouched (rad ~= 4.6deg)
