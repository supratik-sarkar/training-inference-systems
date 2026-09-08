"""Report schema and summary rendering (single process)."""
from __future__ import annotations

import json

from dtp import Report, load_report, summarise
from dtp.experiments import ExperimentResult


def sample() -> Report:
    return Report.build(4, (4,), [
        ExperimentResult("a", True, {"x": 1}),
        ExperimentResult("b", False, error="boom"),
        ExperimentResult("c", False, skipped=True, skip_reason="no hardware"),
    ])


def test_counts_are_separated() -> None:
    r = sample()
    assert (r.passed, r.failed, r.skipped) == (1, 1, 1)
    assert not r.all_passed


def test_skips_do_not_count_as_failures() -> None:
    r = Report.build(2, (2,), [ExperimentResult("c", False, skipped=True)])
    assert r.failed == 0
    assert r.all_passed


def test_environment_records_accelerator_availability() -> None:
    env = sample().environment
    assert env["device"] == "cpu"
    assert "cuda_available" in env and "mps_available" in env
    assert env["torch"]


def test_round_trips_through_disk(tmp_path) -> None:
    p = sample().write(tmp_path / "r.json")
    loaded = load_report(p)
    assert loaded.world_size == 4
    assert [x["name"] for x in loaded.results] == ["a", "b", "c"]
    assert json.loads(p.read_text())["schema_version"] == 1


def test_summary_marks_each_outcome() -> None:
    text = summarise(sample())
    assert "PASS  a" in text
    assert "FAIL  b" in text
    assert "SKIP  c" in text
    assert "no hardware" in text
