# apple-silicon-inference-lab

The mechanics of serving a language model locally: what to admit, what order to run it in, how many at once, and how to measure the result without lying about it.

Most of a serving stack is not the model. Admission control, priority queueing, concurrency limits, streaming and latency statistics are all model-independent, and all of it is tested here against a deterministic backend that emits reproducible tokens on a controllable clock.

## The rule this repository is built around

**A backend advertises what it actually supports. Everything else is skipped with a stated reason.**

```
case                         status     reason
--------------------------------------------------------------------
serial_latency               measured
time_to_first_token          measured
inter_token_latency          measured
concurrent_throughput        measured
prefix_cache_reuse           skipped    deterministic does not implement
                                        prefix_cache; no measurement exists
                                        and none is estimated
speculative_speedup          skipped    deterministic does not implement
                                        speculative_decoding; ...
quantization_comparison      skipped    ...
memory_footprint             skipped    deterministic exposes no memory counters
```

There is no fake paged KV cache here, no fake continuous batching, no fake speculative decoding. Those features are the ones a serving repository is most tempted to claim, and a stub that returns a plausible speedup is worse than an empty capability flag, because it is unfalsifiable from the outside.

There is a test asserting that **no backend in this repository advertises speculative decoding or a prefix cache**, so the claim cannot drift in later.

## Capabilities

| Capability | `deterministic` | `mlx` | `llamacpp` |
|---|---|---|---|
| `streaming` | yes | yes | **no** |
| `token_counts` | yes | yes | yes |
| `concurrency` | yes | no | no |
| `quantization` | no | yes | yes |
| `memory_metrics` | no | yes | no |
| `prefix_cache` | no | no | no |
| `speculative_decoding` | no | no | no |

`llamacpp` declares `streaming: false` deliberately. Its subprocess interface returns output on completion, so per-token timestamps would be *reconstructed after the fact rather than measured*, and a TTFT computed from reconstructed timings is not a TTFT.

`mlx` declares `concurrency: false`. A backend that is serial in reality must not be run in parallel by the scheduler — the resulting numbers would describe contention rather than the model. `Scheduler.effective_concurrency` enforces this, and there is a test for it.

## No result files are committed

MLX requires Apple Silicon. This package was developed on x86 Linux, where `import mlx` fails. **No genuine performance number for this repository could have been produced in that environment, so none exists here.**

Three separate guards, all in code:

1. `BenchmarkReport.is_performance_measurement` is true only when the backend is real *and* the machine is Apple Silicon.
2. `report.write(..., require_real=True)` raises `SyntheticResultRefused` otherwise.
3. `assert_plottable()` refuses to draw a chart from anything else. A chart is the most persuasive artefact a benchmarking repository produces and the easiest to produce dishonestly, so the guard sits in front of the plotting code rather than in a review checklist.

A test asserts that any JSON committed under `results/` is a real measurement — so the directory stays empty until a Mac fills it.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
asil capabilities
asil benchmark --backend deterministic
```

Running the deterministic benchmark prints a disclaimer with the results, because harness timings are not model timings:

```
This file is NOT a performance measurement. It was produced by backend
'deterministic' with is_real_inference=False on x86_64. Timings describe
the harness, not a model.
```

## The Mac gate

```bash
./scripts/benchmark_mac.sh
```

Refuses to run on non-arm64. On Apple Silicon it installs the MLX extras, probes capabilities, runs the suite against a real quantised model and writes `results/mlx-<model>.json` with `--require-real`, which will refuse if anything about the environment is wrong.

Until that passes:

- `implementation_complete: true`
- `sandbox_unit_tests: true`
- `platform_runtime: MAC_VALIDATION_REQUIRED`
- `benchmark: MAC_VALIDATION_REQUIRED`
- `publication_ready: false`

## Statistics

`LatencySummary` always carries its own sample count, and flags itself when the sample is thin:

```json
{"count": 8, "p50_s": 0.031, "p99_s": 0.048,
 "caveat": "8 samples; tail percentiles from this few observations are
            indicative, not reliable"}
```

A p99 from eight requests is not a p99. Rather than suppress the field or pretend otherwise, the summary reports it and says what it is worth.

`tokens_per_second` returns `None` rather than infinity when no time elapsed, and `time_to_first_token_s` returns `None` rather than zero when no token was emitted. A missing measurement is `null`, never a plausible-looking default.

## Limitations

- The deterministic backend is not a model. It exercises the harness; it says nothing about inference.
- MLX and llama.cpp adapters are written and **never executed**.
- No real tokenizer in the fake path; token counts there are a whitespace proxy and labelled as such.
- Concurrency is thread-based. For a real backend holding the GIL, this measures scheduling, not parallel decode.
- No continuous batching, no paged attention, no prefix caching. Not because they are unimportant, but because implementing them convincingly requires a real serving engine.
- Single-machine only, no distributed serving, no load balancing across processes.

## Reproducing the accepted state

```bash
pytest -q && ruff check . && asil benchmark --backend deterministic
```

## Licence

Apache-2.0.
