"""RSL-RL PPO runner configuration for the Microduck velocity_swizzle task.

Same PPO hyperparameters as the stride roller task
(``tasks.velocity_rollers.MicroduckRollersRlCfg``), just a new
experiment/run name.
"""

import dataclasses

from mjlab_microduck.tasks.velocity_rollers import MicroduckRollersRlCfg

MicroduckSwizzleRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    experiment_name="velocity_swizzle",
    run_name="velocity_swizzle",
)
