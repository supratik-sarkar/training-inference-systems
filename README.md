# distributed-training-primitives

A correctness laboratory for PyTorch's `DeviceMesh` / `DTensor` stack, run across local CPU processes.

The question this repository answers is *"does the sharded computation agree with the unsharded one?"* — not *"how fast is it?"* Every experiment builds an unsharded reference, distributes it across a mesh, performs the operation in both worlds, and compares. Agreement is a property of the algorithm, and it can be established honestly on a laptop.

**This is not a performance benchmark.** There is no GPU here, no NCCL, no Triton, no CUDA kernel, no multi-node run, and no throughput claim. `torchrun` launches N processes on one machine over the Gloo backend. Anyone reading this as evidence of accelerator-scale training experience would be reading it wrong, and the results file records the exact environment so that misreading is difficult.

## Why bother without a cluster

Most distributed-training bugs are not performance bugs. They are placement bugs, redistribution bugs, gradient-reduction bugs and checkpoint-shape bugs — all of which reproduce at world size 2 on a CPU. The parts that genuinely need a cluster (communication overlap, memory ceilings, interconnect behaviour) are absent here and are not claimed.

## Experiments

| Name | What it establishes |
|---|---|
| `replicated_correctness` | `Replicate()` holds the full value on every rank |
| `shard0_correctness` | `Shard(0)` splits dim 0 evenly; each local shard equals its slice of the reference; gather round-trips |
| `shard1_correctness` | Sharding a non-leading dimension behaves identically |
| `redistribute_shard_to_replicate` | shard → replicate all-gathers to the reference, and back again |
| `matmul_equivalence` | Row-sharded `A @ B` matches the reference within stated float32 tolerance |
| `elementwise_equivalence` | Local ops match the reference exactly |
| `gradient_equivalence` | Backward through a sharded matmul yields the reference gradient |
| `collectives` | `all_reduce` and `all_gather` match a closed-form expectation |
| `distributed_checkpoint` | A sharded tensor saves and loads back equal to the reference |
| `recovery_from_checkpoint` | After a simulated restart, resumed state continues identically to an uninterrupted run |
| `mismatched_mesh_is_rejected` | A ragged shape splits unevenly without losing or duplicating an element |
| `fsdp2_conditional` | `fully_shard` on CPU/Gloo — reported honestly, or explicitly not demonstrated |

## FSDP2 is conditional, deliberately

`fsdp2_conditional` attempts a real `fully_shard` forward and backward. If the installed build requires an accelerator, the result is recorded as `skipped` with reason `FSDP2_NOT_DEMONSTRATED_ON_THIS_HARDWARE` and the specific exception. Nothing is imitated to fill the gap. Check the generated report to see which happened on your machine — the answer depends on your PyTorch build, not on this code.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # single-process unit tests
./scripts/run_experiments.sh 2           # 2 ranks
./scripts/run_experiments.sh 4           # 4 ranks
python -m dtp.cli --list
```

Under `torchrun` directly:

```bash
torchrun --nproc_per_node=4 -m dtp.cli --out results/report-ws4.json
torchrun --nproc_per_node=2 -m dtp.cli --only shard0_correctness gradient_equivalence
```

Exit status is non-zero if any experiment fails. Skips do not fail the run.

## Results

Rank 0 writes one JSON document per run:

```json
{
  "schema_version": 1,
  "world_size": 4,
  "mesh_shape": [4],
  "backend": "gloo",
  "environment": {
    "torch": "...", "python": "...", "platform": "...",
    "device": "cpu", "cuda_available": false, "mps_available": false
  },
  "results": [
    {"name": "shard0_correctness", "passed": true,
     "detail": {"global_shape": [16, 3], "local_shape": [4, 3],
                "gather_equals_reference": true},
     "error": null, "skipped": false, "skip_reason": ""}
  ]
}
```

`cuda_available` and `mps_available` are recorded so a reader can confirm what the run was and was not.

Committed result files are produced by an actual run. There are no placeholder values in this repository.

## Numerical tolerance

Exact equality is asserted where the operation is bitwise deterministic (replication, sharding, elementwise, gather). Where a reduction order differs between the sharded and reference paths — matmul and gradients — the comparison uses `torch.allclose` with the tolerance recorded in the result detail. Claiming bitwise equality for a reordered float32 reduction would be wrong.

## Limitations

- CPU and Gloo only. No NCCL, no GPU, no multi-node.
- World size is bounded by local cores. Results at world size 2–8 say nothing about world size 512.
- No communication/computation overlap, no memory profiling, no bandwidth measurement.
- Tensor and pipeline parallelism are not implemented. A CPU-process imitation of tensor parallelism would demonstrate nothing that `DTensor` placement does not already show.
- `torch.compile` is not exercised under distribution; it added no correctness signal here.
- The models are a few small `Linear` layers, sized so every experiment finishes in seconds.

## Reproducing the accepted state

```bash
pytest -q && ./scripts/run_experiments.sh 2
```

## Licence

Apache-2.0.
