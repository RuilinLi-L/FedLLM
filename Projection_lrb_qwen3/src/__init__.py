"""Isolated Qwen3 SST-2 preregistration, diagnostics, and none-only DAGER utilities.

Projection-LRB and every defended-gradient path remain intentionally absent.
"""

from .config import ExperimentConfig, PreregistrationConfigError, load_experiment_config

__all__ = [
    "ExperimentConfig",
    "PreregistrationConfigError",
    "load_experiment_config",
]
