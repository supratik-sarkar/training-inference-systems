# results

Empty by design.

No benchmark has ever run in this repository's build environment, because MLX
requires Apple Silicon and no GGUF model is present for llama.cpp. There is no
example results file and no sample chart, because both would be fabrications
in the shape of evidence.

`BenchReport.write()` raises `SyntheticResultRefused` for reports from the
deterministic backend, and `plotting.load_results()` raises on a missing or
synthetic file. The first real file here will be written by
`scripts/benchmark_mac.sh`.
