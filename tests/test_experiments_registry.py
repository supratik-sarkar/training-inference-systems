"""Registry integrity and single-process guards."""
from __future__ import annotations

import pytest

from dtp import EXPERIMENTS, is_distributed
from dtp.experiments import ExperimentResult
from dtp.harness import process_group, seeded


def test_every_experiment_is_callable() -> None:
    assert len(EXPERIMENTS) == 12
    assert all(callable(fn) for fn in EXPERIMENTS.values())


def test_names_match_their_keys() -> None:
    for name, fn in EXPERIMENTS.items():
        assert fn.__name__ == f"exp_{name}"


def test_unknown_experiment_raises() -> None:
    from dtp import run_experiment
    with pytest.raises(KeyError):
        run_experiment("nope", None)  # type: ignore[arg-type]


def test_seed_is_identical_across_calls() -> None:
    import torch
    a = torch.randn(4, generator=seeded(1234))
    b = torch.randn(4, generator=seeded(1234))
    assert torch.equal(a, b)


def test_harness_refuses_outside_torchrun() -> None:
    if is_distributed():
        pytest.skip("running under torchrun")
    with pytest.raises(RuntimeError, match="not launched under torchrun"), process_group():
        pass


def test_result_serialises() -> None:
    d = ExperimentResult("x", True, {"k": 1}).to_dict()
    assert d["name"] == "x" and d["passed"] is True and d["skipped"] is False
