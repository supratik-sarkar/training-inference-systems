# What this repository does not show

Written first, because it is the more important half.

## Not shown

**Accelerator performance.** No GPU is involved. `cuda_available` is recorded
as `false` in every committed result. Nothing here measures FLOPs, memory
bandwidth, kernel occupancy or interconnect throughput.

**Scale.** World size here is bounded by local CPU cores. Behaviour at world
size 4 tells you the algorithm is right; it tells you nothing about world size
512, where the dominant concerns — straggler ranks, communication overlap,
gradient compression, failure rates — do not appear at all.

**Tensor or pipeline parallelism.** Deliberately absent. Simulating TP across
CPU processes would produce a diagram, not a result. `DTensor` placement
already demonstrates the sharding semantics that matter, without pretending.

**Custom kernels.** No Triton, no CUDA C++. Neither is available on this
hardware and neither is imitated.

**Mixed precision at scale.** Everything is float32. FP16/BF16/FP8 numerics
are a hardware question, and a CPU run would misrepresent them.

**Memory-constrained training.** FSDP2's purpose is fitting models that do not
otherwise fit. That property cannot be demonstrated where memory was never the
binding constraint.

## Shown

The correctness properties that a cluster does not change:

- placement semantics: what each rank physically holds under `Replicate()` and
  `Shard(n)`;
- redistribution: that all-gather and re-shard round-trip without loss;
- computation equivalence: that a sharded matmul agrees with its reference,
  within the float32 tolerance recorded alongside the result;
- gradient equivalence: that backward through sharded operands produces the
  reference gradient;
- collective behaviour against a closed-form expectation;
- checkpoint round-trip across ranks;
- restart recovery: that resuming from a checkpoint continues identically to a
  run that was never interrupted.

These are the properties that break silently and are worth testing. They
reproduce faithfully at world size 2 on a laptop, which is the entire argument
for this repository existing.

## On FSDP2

`fsdp2_conditional` attempts a genuine `fully_shard` forward and backward on
CPU/Gloo. If the installed PyTorch build permits it, the result records that
the semantics work and explicitly notes it is not a scaling result. If the
build requires an accelerator, the result is `skipped` with reason
`FSDP2_NOT_DEMONSTRATED_ON_THIS_HARDWARE` and the exception text. There is no
third path in which something is imitated.
