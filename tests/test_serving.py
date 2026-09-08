"""Admission, queueing, scheduling, routing and statistics."""
from __future__ import annotations

import pytest

from asil import (
    AdmissionDecision,
    AdmissionPolicy,
    Capability,
    DeterministicBackend,
    GenerationRequest,
    LlamaCppBackend,
    MLXBackend,
    ModelRouter,
    PriorityQueue,
    Scheduler,
    UnsupportedCapability,
    percentile,
    summarise,
    tokens_per_second,
)


def req(rid: str, prompt: str = "hello world", **kw) -> GenerationRequest:
    return GenerationRequest(rid, prompt, **kw)


# -- admission -------------------------------------------------------------
def test_empty_prompt_is_rejected() -> None:
    r = AdmissionPolicy().admit(req("a", "   "), 0)
    assert r.decision is AdmissionDecision.REJECTED_MALFORMED


def test_non_positive_max_tokens_is_rejected() -> None:
    r = AdmissionPolicy().admit(req("a", max_tokens=0), 0)
    assert r.decision is AdmissionDecision.REJECTED_MALFORMED


def test_overlong_completion_is_rejected() -> None:
    r = AdmissionPolicy(max_completion_tokens=10).admit(req("a", max_tokens=11), 0)
    assert r.decision is AdmissionDecision.REJECTED_TOO_LONG


def test_overlong_prompt_is_rejected() -> None:
    p = AdmissionPolicy(max_prompt_tokens=3)
    assert p.admit(req("a", "one two three four"), 0).decision is (
        AdmissionDecision.REJECTED_TOO_LONG)


def test_full_queue_is_rejected_rather_than_queued() -> None:
    r = AdmissionPolicy(max_queue_depth=2).admit(req("a"), 2)
    assert r.decision is AdmissionDecision.REJECTED_QUEUE_FULL
    assert "queue depth" in r.reason


def test_valid_request_is_accepted() -> None:
    assert AdmissionPolicy().admit(req("a"), 0).accepted


# -- queue -----------------------------------------------------------------
def test_priority_beats_arrival_order() -> None:
    q = PriorityQueue()
    q.push(req("low", priority=0))
    q.push(req("high", priority=5))
    assert q.pop().request_id == "high"


def test_equal_priority_is_fifo_not_arbitrary() -> None:
    q = PriorityQueue()
    for i in range(5):
        q.push(req(f"r{i}", priority=1))
    assert [q.pop().request_id for _ in range(5)] == [f"r{i}" for i in range(5)]


# -- scheduler -------------------------------------------------------------
def test_scheduler_completes_every_admitted_request() -> None:
    s = Scheduler(DeterministicBackend())
    for i in range(6):
        assert s.submit(req(f"r{i}", max_tokens=4)).accepted
    responses = s.drain()
    assert len(responses) == 6
    assert s.stats.completed == 6 and s.stats.failed == 0


def test_scheduler_records_rejections_by_reason() -> None:
    s = Scheduler(DeterministicBackend(), AdmissionPolicy(max_queue_depth=2))
    for i in range(5):
        s.submit(req(f"r{i}", max_tokens=4))
    assert s.stats.admitted == 2
    assert s.stats.rejections["rejected_queue_full"] == 3


def test_backend_failure_is_recorded_not_swallowed() -> None:
    s = Scheduler(DeterministicBackend(fail_on={"bad"}))
    s.submit(req("good", max_tokens=2))
    s.submit(req("bad", max_tokens=2))
    responses = s.drain()
    assert len(responses) == 1
    assert s.stats.failed == 1


def test_non_concurrent_backend_is_run_serially() -> None:
    """Running a serial backend in parallel would measure contention."""
    s = Scheduler(MLXBackend(), max_concurrency=8)
    assert s.effective_concurrency == 1


def test_concurrent_backend_uses_the_configured_width() -> None:
    assert Scheduler(DeterministicBackend(), max_concurrency=4
                     ).effective_concurrency == 4


def test_streaming_is_refused_when_unsupported() -> None:
    s = Scheduler(LlamaCppBackend())
    with pytest.raises(UnsupportedCapability, match="streaming"):
        s.stream_one(req("a"))


# -- routing ---------------------------------------------------------------
def test_router_falls_back_to_default() -> None:
    d = DeterministicBackend()
    router = ModelRouter(default=d)
    assert router.resolve(req("a"))[0] == "default"


def test_router_honours_an_explicit_tag() -> None:
    d, m = DeterministicBackend(), MLXBackend()
    router = ModelRouter(default=d)
    router.add_route("big", m)
    name, backend = router.resolve(req("a", metadata={"route": "big"}))
    assert name == "big" and backend is m


def test_router_explains_its_decision() -> None:
    router = ModelRouter(default=DeterministicBackend())
    e = router.explain(req("a", metadata={"route": "missing"}))
    assert e["route"] == "default"
    assert e["requested_route"] == "missing"
    assert e["is_real_inference"] is False


# -- statistics ------------------------------------------------------------
def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_of_empty_sample_is_none() -> None:
    assert percentile([], 50) is None


def test_percentile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0], 101)


def test_single_sample_percentile() -> None:
    assert percentile([7.0], 99) == 7.0


def test_thin_samples_carry_a_caveat() -> None:
    d = summarise([0.1, 0.2, 0.3]).to_dict()
    assert "caveat" in d
    assert "indicative" in d["caveat"]


def test_large_samples_carry_no_caveat() -> None:
    assert "caveat" not in summarise([0.1] * 40).to_dict()


def test_tokens_per_second_is_none_for_zero_time() -> None:
    assert tokens_per_second(10, 0.0) is None
    assert tokens_per_second(0, 1.0) is None


def test_tokens_per_second_computes() -> None:
    assert tokens_per_second(100, 2.0) == 50.0


# -- response timing -------------------------------------------------------
def test_ttft_is_none_without_tokens() -> None:
    from asil import GenerationResponse
    r = GenerationResponse("a", "", 1, 0, [], 1.0, 2.0)
    assert r.time_to_first_token_s is None


def test_deterministic_backend_produces_stable_output() -> None:
    b = DeterministicBackend(token_delay_s=0, first_token_delay_s=0)
    a = b.generate(req("a", "same prompt", max_tokens=6)).text
    c = b.generate(req("b", "same prompt", max_tokens=6)).text
    assert a == c


def test_deterministic_backend_declares_itself_fake() -> None:
    b = DeterministicBackend()
    assert b.is_real_inference is False
    assert b.capabilities.supports(Capability.STREAMING)
    assert not b.capabilities.supports(Capability.SPECULATIVE_DECODING)
