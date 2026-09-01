# `tasks/mdp.py` → `tasks/mdp/` — status & notes

## What changed

The ~7200-line, 229-symbol `tasks/mdp.py` (every reward/observation/event/
termination/curriculum/command function for every microduck task, in one
file) is now a package, split by MDP manager category:

```
tasks/mdp/
  __init__.py       # re-exports everything, so `microduck_mdp.<name>` is unchanged
  patches.py         # import-time monkey-patches (RewardManager/PPO/ONNX export) -- side effects only, not a term category
  _common.py         # private helpers/constants shared across categories (e.g. _servo_joint_pos, _fallen_mask, _DEFAULT_ASSET_CFG)
  observations.py    # ObservationTermCfg.func targets
  actions.py          # ActionTermCfg -- empty; every task uses mjlab's built-in JointPositionActionCfg directly
  rewards.py          # RewardTermCfg.func targets (147 of the 229 symbols)
  events.py           # EventTermCfg.func targets
  terminations.py     # TerminationTermCfg.func targets
  curriculum.py        # CurriculumTermCfg.func targets
  commands.py          # CommandTermCfg / CommandTerm subclasses (UniformPoseCommand, VelocityCommandCommandOnly, SitStandCommand, GroundPickPhaseCommand, RelativeHeadingVelocityCommand, ...)
```

**No env-cfg file needed to change.** Every task file does
`from mjlab_microduck.tasks import mdp as microduck_mdp` and then
`func=microduck_mdp.some_name` — that import and every one of those
attribute lookups keeps working exactly as before, because `mdp/__init__.py`
re-exports every public name from every submodule into the same flat
namespace. This was verified directly: every one of the 132 distinct
`microduck_mdp.X` attributes actually referenced anywhere in `tasks/*.py` /
`tasks/velocity/**/*.py` is present on the package after import.

## How the split was done (and how it was checked)

This wasn't a manual eyeball-and-copy job — the risk with a mechanical split
like this is silently losing or misplacing code that sits *between* two
functions (stray module-level constants, monkey-patch application code,
banner comments) or breaking a function that calls another function by bare
name across the new file boundary. Both classes of bug were checked for
systematically, not just spot-checked:

1. **Category assignment was derived from real usage, not guessed from
   names.** Every `func=microduck_mdp.X` in every task/env-cfg file was
   matched against the `RewTerm`/`ObsTerm`/`EventTerm`/`DoneTerm`/`CurrTerm`
   wrapping it, giving a ground-truth category for ~150 symbols with zero
   conflicts. The rest (private helpers, command classes, internal-only
   reward-shaping helpers) were categorized by where they're actually called
   from.
2. **Cross-category bare-name calls were checked exhaustively.** A script
   scanned every function body for calls to every other symbol defined in
   the old file; where the callee's category differed from the caller's, that
   would be a `NameError` after the split unless fixed. This caught 7 real
   cases (a handful of `_xxx_error`/`_spin_*` helpers that were initially
   filed under "common" but actually call reward functions) — they were
   reclassified into `rewards.py` to avoid a circular import between
   `_common.py` and `rewards.py`.
3. **Stray top-level constants between functions were checked exhaustively**
   (every bare `NAME = ...` at column 0 in the old file, not just the ones
   that looked suspicious). This caught real bugs from the mechanical
   def/class-boundary slicing: `_DEFAULT_ASSET_CFG`, `_NECK_JOINT_PATTERNS`,
   the `SPIN_PERIOD`..`SPIN_BRAKE_END` block, and 9 roulade/head-latch
   constants had all landed in the wrong file (or been silently dropped
   entirely, in the case of the first two) because they sit in the gap
   between one function's end and the next function's start, and that gap
   was being attributed to the wrong side of a category boundary. All were
   moved to their correct file (mostly `_common.py`, since they're read by
   helpers used from multiple categories).
4. **`pyflakes` was run over every generated file** as a final, independent,
   scope-aware undefined-name check (a hand-rolled AST check was tried first
   and threw hundreds of false positives from not understanding local
   variable scoping — pyflakes doesn't have that problem). The only
   remaining pyflakes findings are pre-existing, unrelated to the split
   (e.g. a couple of functions in the original file assign a local variable
   they never read; `curriculum.py` has two functions with a local
   `from ... import UniformVelocityCommandCfg` shadowing the module-level
   one) and were left untouched, since fixing pre-existing code smells
   wasn't the ask.
5. **A duplicate `pose_target_match` definition** (two functions with the
   same name, ~300 lines apart) existed in the original file; Python
   silently let the second one shadow the first, so the first was dead code.
   Both were preserved verbatim in `rewards.py` in original order, so the
   exact same shadowing (second wins) still happens — a comment now marks
   the first one as dead code inherited from before the split, in case it's
   worth deleting in a follow-up.
6. Isolated-import test (via lightweight `mjlab`/`torch`/`mujoco` stubs, the
   same approach used for the `velocity` env-cfg refactor) confirms
   `mjlab_microduck.tasks.mdp` imports cleanly, all 132 externally-used
   attributes are present, `pose_target_match` resolves to the correct
   (second) definition, and `make_microduck_velocity_env_cfg()` still
   produces the same reward/event/observation/curriculum/command counts as
   before the split (16/13/8/13/7-8/3/4) for all 4 `play`/`rough`
   combinations.

## Not done / known limitations

- A handful of `.md` planning docs under `docs/superpowers/` mention
  `tasks/mdp.py` by its old path in prose — historical documents, not code,
  left as-is.
- Only import-level correctness was verified (no `torch`/`mujoco-warp`
  available in this environment to actually run a training step). Run
  `uv run pytest tests/` and a short `uv run train ... --max_iterations 5`
  smoke run on real hardware before trusting this in production, the same
  recommendation as for the `velocity` env-cfg refactor.
