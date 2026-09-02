"""Tunables for the Microduck velocity_swizzle (classic swizzle) task.

A separate roller task producing a CLASSIC SWIZZLE: both blades stay on the
ground, the legs spread out and pull back in SYMMETRICALLY (hourglass
pattern), propelling the duck forward. Simpler / more stable alternative to
the alternating stride (``Mjlab-Velocity-Flat-MicroDuck-Rollers``), which
does not transfer well to the real robot. The stride env is left untouched.

Approach A (see docs/superpowers/specs/2026-07-23-swizzle-env-design.md):
the base roller recipe NATURALLY converges to a swizzle, so this task
reuses the stride env wholesale (robot, 61D obs, command, full DR,
curricula, sim2real -- deploys identically with ``--roller``) and only
swaps the reward recipe:
  - REMOVE the anti-swizzle / stride terms.
  - ADD leg_symmetry (legs mirror) + grounded (both blades down).
"""

NUM_STEPS_PER_ENV = 24

# Stride / anti-swizzle rewards to drop for the swizzle task.
ANTI_SWIZZLE_REWARDS = ("single_support", "glide", "skating_air_time", "gait_symmetry", "hip_roll_neutral")

# Head-pose command ranges (4D deltas from HOME: [neck_pitch, head_pitch,
# head_yaw, head_roll]) -- shared between the command's initial value and
# the curriculum's stage-0 value, so they can't drift apart.
HEAD_POSE_INITIAL_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
