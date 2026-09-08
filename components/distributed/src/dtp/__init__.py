"""distributed-training-primitives: correctness experiments for DTensor sharding."""
from __future__ import annotations

from .experiments import EXPERIMENTS, ExperimentResult, run_experiment
from .harness import DistContext, distributed_main, is_distributed
from .report import Report, load_report, summarise

__version__ = "0.1.0"

__all__ = ["DistContext", "EXPERIMENTS", "ExperimentResult", "Report",
           "distributed_main", "is_distributed", "load_report",
           "run_experiment", "summarise", "__version__"]
