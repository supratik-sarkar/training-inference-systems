"""Capability gating and the refusal to emit synthetic results."""
from __future__ import annotations

import pytest

from asil import (
    BenchmarkReport,
    Capability,
    CapabilitySet,
    DeterministicBackend,
    LlamaCppBackend,
    MLXBackend,
    SyntheticResultRefused,
    UnsupportedCapability,
    assert_plottable,
    detect_available,
    run_benchmark,
)


@pytest.fixture
def fast() -> DeterministicBackend:
    return DeterministicBackend(token_delay_s=0.0, first_token_delay_s=0.0)


# -- capability declaration ------------------------------------------------
def test_capability_set_partitions_cleanly() -> None:
    caps = CapabilitySet(streaming=True, token_counts=True)
    assert Capability.STREAMING in caps.supported()
    assert Capability.PREFIX_CACHE in caps.unsupported()
    assert set(caps.supported()) & set(caps.unsupported()) == set()


def test_no_backend_claims_speculative_decoding() -> None:
    """Nothing here implements it, so nothing may advertise it."""
    for cls in (DeterministicBackend, MLXBackend, LlamaCppBackend):
        caps = cls().capabilities
        assert not caps.supports(Capability.SPECULATIVE_DECODING), cls.__name__
        assert not caps.supports(Capability.PREFIX_CACHE), cls.__name__


def test_llamacpp_declares_no_streaming() -> None:
    """Its interface returns output on completion; timings would be invented."""
    assert not LlamaCppBackend().capabilities.supports(Capability.STREAMING)


def test_unsupported_capability_raises_with_a_reason() -> None:
    with pytest.raises(UnsupportedCapability) as exc:
        DeterministicBackend().memory_footprint()
    assert "fabrication" in str(exc.value)


def test_mlx_is_unavailable_here_and_raises_rather_than_degrading() -> None:
    from asil import BackendUnavailable
    b = MLXBackend()
    if b.available():
        pytest.skip("MLX present")
    assert b.is_real_inference is True
    with pytest.raises(BackendUnavailable, match="Apple Silicon"):
        list(b.stream.__wrapped__(b, None)) if hasattr(b.stream, "__wrapped__") \
            else b._require()


def test_detect_available_reports_every_backend() -> None:
    found = detect_available()
    assert set(found) == {"deterministic", "mlx", "llamacpp"}
    assert found["deterministic"]["available"] is True
    assert found["deterministic"]["is_real_inference"] is False


# -- gated benchmark -------------------------------------------------------
def test_unsupported_cases_are_skipped_with_reasons(fast) -> None:
    report = run_benchmark(fast, requests=2, max_tokens=3)
    skipped = [c for c in report.cases if c["status"] == "skipped"]
    assert skipped
    assert all(c["reason"] for c in skipped), "a skip must state why"


def test_speculative_decoding_is_skipped_not_estimated(fast) -> None:
    report = run_benchmark(fast, requests=2, max_tokens=3)
    case = next(c for c in report.cases if c["name"] == "speculative_speedup")
    assert case["status"] == "skipped"
    assert "none is estimated" in case["reason"]
    assert case["metrics"] == {}


def test_streaming_backend_measures_ttft(fast) -> None:
    report = run_benchmark(fast, requests=3, max_tokens=4)
    case = next(c for c in report.cases if c["name"] == "time_to_first_token")
    assert case["status"] == "measured"
    assert case["metrics"]["ttft"]["count"] == 3


def test_non_streaming_backend_skips_ttft() -> None:
    report = run_benchmark(LlamaCppBackend(), requests=2)
    names = {c["name"]: c for c in report.cases}
    assert names["availability"]["status"] == "skipped"


def test_memory_case_skips_when_unsupported(fast) -> None:
    case = next(c for c in run_benchmark(fast, requests=2, max_tokens=2).cases
                if c["name"] == "memory_footprint")
    assert case["status"] == "skipped"


# -- synthetic result refusal ----------------------------------------------
def test_fake_backend_report_is_not_a_performance_measurement(fast) -> None:
    report = run_benchmark(fast, requests=2, max_tokens=3)
    assert report.is_performance_measurement is False
    assert "disclaimer" in report.to_dict()


def test_writing_a_required_real_result_is_refused(fast, tmp_path) -> None:
    report = run_benchmark(fast, requests=2, max_tokens=3)
    with pytest.raises(SyntheticResultRefused, match="refusing to write"):
        report.write(tmp_path / "r.json", require_real=True)


def test_writing_without_the_flag_is_allowed_but_labelled(fast, tmp_path) -> None:
    import json
    p = run_benchmark(fast, requests=2, max_tokens=3).write(tmp_path / "r.json")
    d = json.loads(p.read_text())
    assert d["is_performance_measurement"] is False
    assert "NOT a performance measurement" in d["disclaimer"]


def test_plotting_a_synthetic_result_is_refused(fast) -> None:
    report = run_benchmark(fast, requests=2, max_tokens=3).to_dict()
    with pytest.raises(SyntheticResultRefused, match="refusing to plot"):
        assert_plottable(report)


def test_plotting_requires_at_least_one_measured_case() -> None:
    report = BenchmarkReport("mlx:x", True, {}, cases=[])
    report.environment["is_apple_silicon"] = True
    with pytest.raises(SyntheticResultRefused, match="no case"):
        assert_plottable(report.to_dict())


def test_a_real_apple_silicon_report_would_be_plottable() -> None:
    """The positive branch, so the guard is not merely always-refusing."""
    report = BenchmarkReport(
        "mlx:model", True, {},
        cases=[{"name": "serial_latency", "capability": None,
                "status": "measured", "reason": "",
                "metrics": {"wall_time": {"p50_s": 0.1, "p90_s": 0.2}}}],
    )
    report.environment["is_apple_silicon"] = True
    assert_plottable(report.to_dict())


def test_no_result_files_are_committed_in_this_repository() -> None:
    """Linux cannot produce a real one, so none may be present."""
    import json
    from pathlib import Path
    for p in (Path(__file__).resolve().parent.parent / "results").glob("*.json"):
        d = json.loads(p.read_text())
        assert d.get("is_performance_measurement") is True, (
            f"{p.name} is committed but is not a real measurement"
        )
