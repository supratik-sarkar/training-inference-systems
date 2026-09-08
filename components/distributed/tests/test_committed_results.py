"""Committed result files must be real, complete and self-describing.

This is the guard against placeholder results reaching the repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = sorted(Path(__file__).resolve().parent.parent.glob("results/*.json"))


def test_results_exist() -> None:
    assert RESULTS, "no result files committed; run scripts/run_experiments.sh"


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_result_file_is_well_formed(path: Path) -> None:
    d = json.loads(path.read_text())
    assert d["schema_version"] == 1
    assert d["backend"] == "gloo"
    assert d["world_size"] >= 2
    assert d["environment"]["device"] == "cpu"
    assert d["environment"]["torch"]
    assert d["generated_at"]


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_no_experiment_failed(path: Path) -> None:
    d = json.loads(path.read_text())
    failed = [r["name"] for r in d["results"]
              if not r["passed"] and not r["skipped"]]
    assert not failed, f"{path.name} has failures: {failed}"


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_results_are_not_placeholders(path: Path) -> None:
    d = json.loads(path.read_text())
    assert len(d["results"]) == 12
    blob = json.dumps(d).lower()
    for token in ("todo", "placeholder", "tbd", "xxx", "fixme", "example.com"):
        assert token not in blob, f"placeholder token {token!r} in {path.name}"


BANNED = ("nccl", "cuda kernel", "tensor parallel", "multi-node",
          "throughput", "tokens/s", "gpu")


def _affirmative_surfaces(d: dict) -> list[str]:
    """Everything except free-text notes.

    Notes are handled separately because a note may legitimately mention a
    banned term in order to disclaim it -- "this is not a throughput result"
    is exactly the sentence this repository should contain.
    """
    out: list[str] = []
    for r in d["results"]:
        out.append(r["name"])
        out.append(r.get("skip_reason") or "")
        for k, v in (r.get("detail") or {}).items():
            if k == "note":
                continue
            out.append(f"{k}={v}")
    return out


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_no_affirmative_accelerator_claim(path: Path) -> None:
    """A CPU run must not carry a result asserting accelerator execution."""
    d = json.loads(path.read_text())
    assert d["environment"]["cuda_available"] is False
    blob = " ".join(_affirmative_surfaces(d)).lower()
    for banned in BANNED:
        assert banned not in blob, f"{banned!r} asserted in a CPU result file"


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_notes_mentioning_scale_terms_are_disclaimers(path: Path) -> None:
    """If a note names a scale concept, it must be denying it, not claiming it."""
    d = json.loads(path.read_text())
    for r in d["results"]:
        note = ((r.get("detail") or {}).get("note") or "").lower()
        if not note:
            continue
        for banned in BANNED:
            if banned in note:
                assert "not " in note or "never" in note, (
                    f"{r['name']} note mentions {banned!r} without denying it: "
                    f"{note!r}"
                )
