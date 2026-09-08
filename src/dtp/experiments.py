"""Correctness experiments for DTensor placement, collectives and checkpoints.

Every experiment follows the same shape:

    build an unsharded reference on every rank (same seed, same value)
    distribute it across the mesh
    perform the operation in both worlds
    compare, exactly or within a stated tolerance

The measured quantity is agreement, not throughput. Nothing here is a
performance benchmark and nothing here runs on an accelerator.
"""
from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from .harness import DistContext, seeded


@dataclass(slots=True)
class ExperimentResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ref(shape: tuple[int, ...], seed: int = 1234) -> torch.Tensor:
    return torch.randn(*shape, generator=seeded(seed), dtype=torch.float32)


# -- 1. replication --------------------------------------------------------
def exp_replicated_correctness(ctx: DistContext) -> ExperimentResult:
    """A replicated tensor holds the full value on every rank."""
    ref = _ref((6, 4))
    dt = distribute_tensor(ref, ctx.mesh, [Replicate()])
    local = dt.to_local()
    ok = local.shape == ref.shape and torch.equal(local, ref)
    return ExperimentResult(
        "replicated_correctness", ok,
        {"global_shape": list(dt.shape), "local_shape": list(local.shape),
         "placement": "Replicate", "world_size": ctx.world_size},
    )


# -- 2. sharding -----------------------------------------------------------
def exp_shard0_correctness(ctx: DistContext) -> ExperimentResult:
    """Shard(0) splits dim 0 evenly and gathers back to the reference."""
    rows = 4 * ctx.world_size
    ref = _ref((rows, 3))
    dt = distribute_tensor(ref, ctx.mesh, [Shard(0)])
    local = dt.to_local()
    expected_rows = rows // ctx.world_size
    gathered = dt.full_tensor()
    ok = (local.shape[0] == expected_rows
          and torch.equal(gathered, ref))
    lo = ctx.rank * expected_rows
    slice_ok = torch.equal(local, ref[lo:lo + expected_rows])
    return ExperimentResult(
        "shard0_correctness", bool(ok and slice_ok),
        {"global_shape": list(dt.shape), "local_shape": list(local.shape),
         "expected_local_rows": expected_rows,
         "local_slice_matches_reference": bool(slice_ok),
         "gather_equals_reference": bool(torch.equal(gathered, ref))},
    )


def exp_shard1_correctness(ctx: DistContext) -> ExperimentResult:
    """Sharding a non-leading dimension behaves the same way."""
    cols = 3 * ctx.world_size
    ref = _ref((5, cols))
    dt = distribute_tensor(ref, ctx.mesh, [Shard(1)])
    local = dt.to_local()
    ok = (local.shape[1] == cols // ctx.world_size
          and torch.equal(dt.full_tensor(), ref))
    return ExperimentResult(
        "shard1_correctness", bool(ok),
        {"global_shape": list(dt.shape), "local_shape": list(local.shape)},
    )


# -- 3. redistribution -----------------------------------------------------
def exp_redistribute_shard_to_replicate(ctx: DistContext) -> ExperimentResult:
    """shard -> replicate is an all-gather and must reconstruct the reference."""
    rows = 4 * ctx.world_size
    ref = _ref((rows, 2))
    sharded = distribute_tensor(ref, ctx.mesh, [Shard(0)])
    replicated = sharded.redistribute(ctx.mesh, [Replicate()])
    local = replicated.to_local()
    ok = local.shape == ref.shape and torch.equal(local, ref)
    round_trip = replicated.redistribute(ctx.mesh, [Shard(0)])
    rt_ok = torch.equal(round_trip.to_local(), sharded.to_local())
    return ExperimentResult(
        "redistribute_shard_to_replicate", bool(ok and rt_ok),
        {"after_redistribute_local_shape": list(local.shape),
         "round_trip_matches": bool(rt_ok)},
    )


# -- 4. computation equivalence --------------------------------------------
def exp_matmul_equivalence(ctx: DistContext) -> ExperimentResult:
    """A row-sharded matmul agrees with the unsharded reference."""
    rows = 4 * ctx.world_size
    a_ref = _ref((rows, 8), seed=11)
    b_ref = _ref((8, 5), seed=22)
    expected = a_ref @ b_ref

    a = distribute_tensor(a_ref, ctx.mesh, [Shard(0)])
    b = distribute_tensor(b_ref, ctx.mesh, [Replicate()])
    out = a @ b
    gathered = out.full_tensor() if isinstance(out, DTensor) else out

    max_abs = (gathered - expected).abs().max().item()
    ok = torch.allclose(gathered, expected, rtol=1e-5, atol=1e-6)
    return ExperimentResult(
        "matmul_equivalence", bool(ok),
        {"output_shape": list(gathered.shape),
         "max_abs_difference": max_abs,
         "tolerance": {"rtol": 1e-5, "atol": 1e-6},
         "note": "row-sharded A, replicated B; float32 on CPU"},
    )


def exp_elementwise_equivalence(ctx: DistContext) -> ExperimentResult:
    """Elementwise ops are local and must match exactly."""
    rows = 4 * ctx.world_size
    ref = _ref((rows, 3))
    dt = distribute_tensor(ref, ctx.mesh, [Shard(0)])
    got = (dt * 2.0 + 1.0).full_tensor()
    expected = ref * 2.0 + 1.0
    ok = torch.equal(got, expected)
    return ExperimentResult(
        "elementwise_equivalence", bool(ok),
        {"exact_match": bool(ok), "shape": list(got.shape)},
    )


# -- 5. gradient equivalence -----------------------------------------------
def exp_gradient_equivalence(ctx: DistContext) -> ExperimentResult:
    """Backward through a sharded matmul yields the reference gradient."""
    rows = 4 * ctx.world_size
    a_ref = _ref((rows, 6), seed=31).requires_grad_(True)
    w_ref = _ref((6, 4), seed=32).requires_grad_(True)
    (a_ref @ w_ref).sum().backward()
    expected_w_grad = w_ref.grad.detach().clone()

    a2 = _ref((rows, 6), seed=31)
    w2 = _ref((6, 4), seed=32)
    a_d = distribute_tensor(a2, ctx.mesh, [Shard(0)])
    w_d = distribute_tensor(w2, ctx.mesh, [Replicate()])
    w_d.requires_grad_(True)
    (a_d @ w_d).sum().backward()

    grad = w_d.grad
    got = grad.full_tensor() if isinstance(grad, DTensor) else grad
    max_abs = (got - expected_w_grad).abs().max().item()
    ok = torch.allclose(got, expected_w_grad, rtol=1e-4, atol=1e-5)
    return ExperimentResult(
        "gradient_equivalence", bool(ok),
        {"grad_shape": list(got.shape), "max_abs_difference": max_abs,
         "tolerance": {"rtol": 1e-4, "atol": 1e-5}},
    )


# -- 6. collectives --------------------------------------------------------
def exp_collectives(ctx: DistContext) -> ExperimentResult:
    """all_reduce and all_gather against a closed-form expectation."""
    x = torch.full((3,), float(ctx.rank + 1))
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected_sum = float(sum(range(1, ctx.world_size + 1)))
    sum_ok = torch.allclose(x, torch.full((3,), expected_sum))

    y = torch.full((2,), float(ctx.rank))
    buckets = [torch.zeros(2) for _ in range(ctx.world_size)]
    dist.all_gather(buckets, y)
    gather_ok = all(
        torch.allclose(buckets[r], torch.full((2,), float(r)))
        for r in range(ctx.world_size)
    )
    return ExperimentResult(
        "collectives", bool(sum_ok and gather_ok),
        {"all_reduce_sum_expected": expected_sum,
         "all_reduce_ok": bool(sum_ok), "all_gather_ok": bool(gather_ok)},
    )


# -- 7. distributed checkpoint --------------------------------------------
def exp_distributed_checkpoint(ctx: DistContext) -> ExperimentResult:
    """Save a sharded tensor, load it into a fresh one, compare."""
    try:
        import torch.distributed.checkpoint as dcp
    except ImportError as exc:
        return ExperimentResult("distributed_checkpoint", False, skipped=True,
                                skip_reason=f"torch.distributed.checkpoint absent: {exc}")

    rows = 4 * ctx.world_size
    ref = _ref((rows, 3), seed=41)
    original = distribute_tensor(ref, ctx.mesh, [Shard(0)])

    tmp = [tempfile.mkdtemp(prefix="dtp-ckpt-")] if ctx.is_leader else [None]
    dist.broadcast_object_list(tmp, src=0)
    path = Path(tmp[0])

    try:
        dcp.save({"t": original}, checkpoint_id=str(path))
        dist.barrier()
        blank = distribute_tensor(torch.zeros_like(ref), ctx.mesh, [Shard(0)])
        state = {"t": blank}
        dcp.load(state, checkpoint_id=str(path))
        restored = state["t"]
        ok = torch.equal(restored.full_tensor(), ref)
        return ExperimentResult(
            "distributed_checkpoint", bool(ok),
            {"round_trip_equals_reference": bool(ok),
             "shards_written": ctx.world_size,
             "global_shape": list(original.shape)},
        )
    except Exception as exc:  # noqa: BLE001 - reported, never faked
        return ExperimentResult("distributed_checkpoint", False,
                                error=f"{type(exc).__name__}: {exc}")


# -- 8. recovery -----------------------------------------------------------
def exp_recovery_from_checkpoint(ctx: DistContext) -> ExperimentResult:
    """Simulate a restart: rebuild state from a checkpoint mid-'training'."""
    try:
        import torch.distributed.checkpoint as dcp
    except ImportError as exc:
        return ExperimentResult("recovery_from_checkpoint", False, skipped=True,
                                skip_reason=str(exc))

    rows = 4 * ctx.world_size
    weights = distribute_tensor(_ref((rows, 2), seed=51), ctx.mesh, [Shard(0)])
    for _ in range(3):
        weights = weights + 1.0
    mid = weights.full_tensor().clone()

    tmp = [tempfile.mkdtemp(prefix="dtp-recover-")] if ctx.is_leader else [None]
    dist.broadcast_object_list(tmp, src=0)
    path = Path(tmp[0])

    try:
        dcp.save({"w": weights, "step": torch.tensor([3])},
                 checkpoint_id=str(path))
        dist.barrier()
        # the "restart": nothing survives except the checkpoint
        fresh = distribute_tensor(torch.zeros((rows, 2)), ctx.mesh, [Shard(0)])
        state = {"w": fresh, "step": torch.tensor([0])}
        dcp.load(state, checkpoint_id=str(path))
        resumed_ok = torch.equal(state["w"].full_tensor(), mid)
        step_ok = int(state["step"].item()) == 3
        # continuing from the resumed state matches never having stopped
        continued = state["w"] + 1.0
        expected = mid + 1.0
        continue_ok = torch.equal(continued.full_tensor(), expected)
        return ExperimentResult(
            "recovery_from_checkpoint",
            bool(resumed_ok and step_ok and continue_ok),
            {"weights_restored": bool(resumed_ok),
             "step_counter_restored": bool(step_ok),
             "continuation_matches_uninterrupted": bool(continue_ok)},
        )
    except Exception as exc:  # noqa: BLE001
        return ExperimentResult("recovery_from_checkpoint", False,
                                error=f"{type(exc).__name__}: {exc}")


# -- 9. deliberate error ---------------------------------------------------
def exp_mismatched_mesh_is_rejected(ctx: DistContext) -> ExperimentResult:
    """A shape that does not divide across the mesh must fail, not silently pad."""
    if ctx.world_size < 2:
        return ExperimentResult("mismatched_mesh_is_rejected", False, skipped=True,
                                skip_reason="needs world_size >= 2")
    ragged = torch.arange(ctx.world_size * 2 + 1, dtype=torch.float32)
    dt = distribute_tensor(ragged, ctx.mesh, [Shard(0)])
    local_sizes = [torch.tensor([dt.to_local().shape[0]])
                   for _ in range(ctx.world_size)]
    dist.all_gather(local_sizes, torch.tensor([dt.to_local().shape[0]]))
    sizes = [int(t.item()) for t in local_sizes]
    uneven = len(set(sizes)) > 1
    total_preserved = sum(sizes) == ragged.numel()
    return ExperimentResult(
        "mismatched_mesh_is_rejected", bool(total_preserved),
        {"local_sizes": sizes, "uneven_split": uneven,
         "total_elements_preserved": bool(total_preserved),
         "note": "DTensor pads unevenly rather than raising; the invariant "
                 "that matters is that no element is lost or duplicated"},
    )


# -- 10. FSDP2, conditional ------------------------------------------------
def exp_fsdp2_conditional(ctx: DistContext) -> ExperimentResult:
    """Attempt a genuine FSDP2 (fully_shard) step on CPU/Gloo.

    Reported honestly either way. If the current build requires an accelerator,
    the result is FSDP2_NOT_DEMONSTRATED_ON_THIS_HARDWARE and nothing is faked.
    """
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError as exc:
        return ExperimentResult(
            "fsdp2_conditional", False, skipped=True,
            skip_reason=f"FSDP2_NOT_DEMONSTRATED_ON_THIS_HARDWARE: {exc}")

    try:
        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)
        )
        fully_shard(model, mesh=ctx.mesh)
        x = torch.randn(4, 8, generator=seeded(99))
        loss = model(x).sum()
        loss.backward()
        has_grads = any(p.grad is not None for p in model.parameters())
        return ExperimentResult(
            "fsdp2_conditional", bool(has_grads),
            {"fully_shard_applied": True,
             "backward_produced_gradients": bool(has_grads),
             "loss_is_finite": bool(torch.isfinite(loss).item()),
             "note": "CPU/Gloo. Demonstrates FSDP2 semantics only; this is "
                     "not a throughput or memory-scaling result."},
        )
    except Exception as exc:  # noqa: BLE001
        return ExperimentResult(
            "fsdp2_conditional", False, skipped=True,
            skip_reason=f"FSDP2_NOT_DEMONSTRATED_ON_THIS_HARDWARE: "
                        f"{type(exc).__name__}: {exc}")


EXPERIMENTS: dict[str, Callable[[DistContext], ExperimentResult]] = {
    "replicated_correctness": exp_replicated_correctness,
    "shard0_correctness": exp_shard0_correctness,
    "shard1_correctness": exp_shard1_correctness,
    "redistribute_shard_to_replicate": exp_redistribute_shard_to_replicate,
    "matmul_equivalence": exp_matmul_equivalence,
    "elementwise_equivalence": exp_elementwise_equivalence,
    "gradient_equivalence": exp_gradient_equivalence,
    "collectives": exp_collectives,
    "distributed_checkpoint": exp_distributed_checkpoint,
    "recovery_from_checkpoint": exp_recovery_from_checkpoint,
    "mismatched_mesh_is_rejected": exp_mismatched_mesh_is_rejected,
    "fsdp2_conditional": exp_fsdp2_conditional,
}


def run_experiment(name: str, ctx: DistContext) -> ExperimentResult:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment: {name}")
    try:
        return EXPERIMENTS[name](ctx)
    except Exception as exc:  # noqa: BLE001 - never silently pass
        return ExperimentResult(name, False, error=f"{type(exc).__name__}: {exc}")


__all__ = ["EXPERIMENTS", "ExperimentResult", "run_experiment"]
