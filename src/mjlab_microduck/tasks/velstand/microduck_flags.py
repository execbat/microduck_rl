"""Tunables for the Microduck velstand (walk + fall recovery) task.

Direct port of the module-level constants from the old
``tasks/microduck_velstand_env_cfg.py``. See that file's original docstring
history (kept verbatim on ``MicroduckVelstandEnvCfg``) for the run-1 through
run-7 design rationale -- this is the most iterated-on task in the repo.
"""

# Phase boundaries (PPO iterations; env step counter scales by num_steps_per_env=24).
FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

# Fallen gates. LESSON (first rebase training run): the recovery REWARDS
# must gate on TILT ONLY. Gating them on low height too made SITTING
# (z~=0.07, trunk upright) open the gate -> the policy learned to sit and
# farm upright_linear while bobbing for com_upward_velocity and shaking its
# legs through the air_time window. Gating a positive reward on a bad state
# rewards entering the state. Tilt>40deg can't be farmed from a comfortable
# pose -- you're genuinely toppled. The TERMINATION keeps the z-condition so
# sitters and stuck-low envs get recycled (terminated) rather than paid.
REWARD_GATE_TILT_DEG = 40.0  # recovery rewards: fallen = tilt > 40deg ONLY
# TERM z-gate at 0.08, NOT 0.10 (run-3 lesson): a normally wobbling upright
# robot dips to z=0.084-0.096 -- 0.10 sits inside the early-learning
# envelope and recycled crouch-walking explorers every 5s. 0.08 still
# catches sitting (z~=0.07) and prone (z~=0.05).
TERM_GATE_Z = 0.08  # fallen_too_long: z < 0.08 OR tilt > 40deg
TERM_GATE_TILT_DEG = 40.0

# "Recovery COMPLETE" definition -- shared by the recovery_success bounty
# and the fallen_tax release (run-5 crouch-endpoint lesson). z threshold
# must sit INSIDE the policy's real standing envelope: run 3 measured a
# normally wobbling upright robot at z ~= 0.084-0.096, and the full STAND
# keyframe settles at ~= 0.117. The old up_z=0.105 demanded standing TALLER
# than the policy ever is in practice -> the bounty never fired ->
# recoveries converged to a deep crouch just past the 40deg gates (where
# every dense recovery term stops paying) instead of finishing the stand.
# 0.09 is reachable every stand yet still 2cm above sitting (z~=0.07) and
# 4cm above prone (z~=0.05).
RECOVERED_UP_TILT_DEG = 25.0
RECOVERED_UP_Z = 0.09

# The tax and bounty exist FOR THE RECOVERY PHASE. Run-3 lesson: fallen_tax
# active from step 0 (dense, -0.5) taught "avoid tilt at all costs" within
# ~25 iters -> crouch-freeze local optimum before walking could bootstrap
# (ep_len pinned at the 5s recycle, air_time never grew). Run-6 tried 800
# ("walk is stable by ~750") and prone recovery never bootstrapped -- 1200
# was never about the walk; it bought a TAX-FREE window (fell_over off at
# 500 -> econ on at 1200) where natural-fall get-up attempts cost nothing
# and the dense progress terms alone could teach them. Run-7 restores it.
RECOVERY_ECON_KICKIN_ITER = 1200

# Failed-recovery backstop: continuously fallen this long -> terminate/reset.
# Run-6: 5s -> 8s. At 5s a face-down recovery spent most of its budget
# getting TO the deep crouch and was recycled right at the frontier --
# almost no on-policy data for the crouch->stand last mile.
FALLEN_TIMEOUT_S = 8.0

# Prone + crouch init ramp (phase 3). Prone capped at 45% (was 2/3 --
# starved the walk); face-down first (easier recovery), face-up mixed in
# later. Run-6: crouch_prob adds a REVERSE-CURRICULUM slice -- envs reset
# directly into random mid-recovery crouches (see set_random_crouch_state)
# so the last mile gets dense data instead of only being reached at the
# tail of rare good rollouts. Run-7: back to the run-5 prone schedule
# (prone AFTER econ, which is AFTER a tax-free natural-fall window); run 6
# started prone+econ together at 800 and prone recovery never
# bootstrapped. Crouch slice alone starts at 800: near-upright states,
# tax-free until econ, and it doubles as full-stand posture data.
PRONE_RAMP_STAGES = [
    {"step": 0, "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.00}},
    {"step": 800 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.15}},
    {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "crouch_prob": 0.15}},
    {"step": 2000 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "crouch_prob": 0.15}},
    {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "crouch_prob": 0.15}},
]
