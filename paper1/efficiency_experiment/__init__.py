"""Configurable known-graph efficiency experiments for paper 1.

Use ``python -m paper1.efficiency_experiment CONFIG.toml``. Programmatic users can
register an estimator, construct ExperimentConfig, then call run_experiment.
"""
from .estimators import EstimatorConfig, available_estimators, fit_estimate, register_estimator
from .schema import ExperimentConfig, from_dict, load_config

__all__ = ["EstimatorConfig", "ExperimentConfig", "available_estimators", "fit_estimate", "register_estimator",
           "from_dict", "load_config", "run_experiment"]


def run_experiment(*args, **kwargs):
    # Keep package imports light and avoid a runpy warning for python -m ...run.
    from .run import run_experiment as execute
    return execute(*args, **kwargs)
