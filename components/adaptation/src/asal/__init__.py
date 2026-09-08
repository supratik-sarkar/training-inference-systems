"""apple-silicon-adapter-lab: dataset, adapter and evaluation lifecycle."""
from __future__ import annotations

from .adapters import (
    AdapterNotFound,
    AdapterNotServable,
    AdapterRecord,
    AdapterRegistry,
    AdapterRouter,
    AdapterStatus,
    TrainingConfig,
    TrainingMethod,
)
from .dataset import SCHEMA_VERSION, Dataset, Example, seed_dataset
from .evaluation import (
    EvalReport,
    FailureCluster,
    Prediction,
    RegressionResult,
    SyntheticEvaluationRefused,
    build_next_dataset,
    compare,
    evaluate,
    mine_failures,
)
from .training import (
    BackendUnavailable,
    FakeTrainingBackend,
    MLXTrainingBackend,
    TrainingBackend,
    TrainingResult,
    capability_report,
)

__version__ = "0.1.0"

__all__ = [
    "AdapterNotFound", "AdapterNotServable", "AdapterRecord",
    "AdapterRegistry", "AdapterRouter", "AdapterStatus", "BackendUnavailable",
    "Dataset", "EvalReport", "Example", "FailureCluster",
    "FakeTrainingBackend", "MLXTrainingBackend", "Prediction",
    "RegressionResult", "SCHEMA_VERSION", "SyntheticEvaluationRefused",
    "TrainingBackend", "TrainingConfig", "TrainingMethod", "TrainingResult",
    "__version__", "build_next_dataset", "capability_report", "compare",
    "evaluate", "mine_failures", "seed_dataset",
]
