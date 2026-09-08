# training-inference-systems

A unified machine learning systems engineering repository spanning early graph neural network acceleration and modern Apple Silicon inference, parameter-efficient adaptation, and distributed training semantics.

---

## Architectural Chronology & Provenance

This repository explicitly delineates historical 2025 graph systems research from modern 2026 local and distributed ML infrastructure:

```
[2025 Historical Foundations: legacy/]
├── legacy/gnn-mixed-precision/      Adaptive mixed-precision execution for large-scale GNNs
└── legacy/gnn-push-pull/            Push-pull batching and communication reduction in graph training
                                                │
                                                ▼ (Evolution into ML execution engines)
[2026 Contemporary Systems: components/]
├── components/inference/            Apple Silicon native inference lab (MLX / PyTorch MPS)
├── components/adaptation/           Apple Silicon parameter-efficient fine-tuning (LoRA)
└── components/distributed/          Distributed training semantics, collectives & gradient primitives
```

> **Note on Provenance**: Historical commit dates establish the dates of the legacy components only; the 2026 systems were added in separately dated consolidation/import commits. Complete Git provenance has been preserved for all components without squashing.

---

## Systems Architecture

### Contemporary 2026 Systems (`components/`)
1. **Apple Silicon Inference Lab (`components/inference/`)**:
   Empirical benchmarking and runtime engine evaluating local autoregressive model execution (Llama-3.2-1B-Instruct) across MLX and PyTorch MPS backends. Measures real TTFT (Time-To-First-Token), decode throughput (tokens/sec), and peak memory utilization under authoritative hardware telemetry.
2. **Apple Silicon Adapter Lab (`components/adaptation/`)**:
   Local parameter-efficient fine-tuning (PEFT) framework providing rank-stabilized Low-Rank Adaptation (LoRA), gradient checkpointing, and memory profiling on unified memory architectures.
3. **Distributed Training Primitives (`components/distributed/`)**:
   Foundational distributed training primitives implementing ring all-reduce, tensor parallel splitting, pipeline communication schedules, and deterministic gradient verification.

### Historical 2025 Engineering (`legacy/`)
1. **Adaptive Mixed Precision GNN (`legacy/gnn-mixed-precision/`)**:
   Early research exploring dynamic numerical precision scaling across heterogeneous node neighborhoods in graph convolution networks.
2. **Push-Pull Graph Batching (`legacy/gnn-push-pull/`)**:
   Decoupled communication/computation batching schedules minimizing node replication overhead in distributed graph embeddings.

---

## Scientific Boundaries & Verification Guarantees

To maintain empirical and scientific integrity, the systems in this repository observe explicit boundaries:
* **Adapter Verification**: Standard LoRA is genuinely implemented, tested, and empirically verified. QLoRA (quantized LoRA) remains unsupported/unverified on the tested macOS toolchain and is explicitly not claimed.
* **Inference Claims**: Inference benchmarks report measurements for genuinely executed configurations only. Unexecuted scenarios—such as concurrent multi-tenant throughput, speculative decoding speedups, and prefix-cache reuse—are intentionally omitted or marked skipped rather than estimated.
* **Distributed Scaling**: Distributed training results represent algorithmic correctness and protocol semantic conformance (tensor shape invariance, all-reduce numerical equivalence), not physical multi-node cluster scaling claims.

---

## Test Baseline & Verification

The active 2026 component test baseline verifies against Python 3.12.13:

| Component | Directory | Baseline Tests | Status |
| :--- | :--- | :---: | :---: |
| Apple Silicon Inference Lab | `components/inference` | 46 | Verified |
| Apple Silicon Adapter Lab | `components/adaptation` | 46 | Verified |
| Distributed Training Primitives | `components/distributed` | 27 | Verified |
| **Total Baseline** | **Active Components** | **119 tests** | **Passing** |

### Running the Ephemeral Verification Suite

The repository includes a standalone verification script that provisions ephemeral virtual environments under `mktemp -d`:

```bash
./scripts/verify_all.sh
```

The script executes:
1. Strict Python 3.12.13 version confirmation (fails closed on mismatch);
2. Ephemeral virtual environment creation in temporary directories;
3. Subsystem dependency installation;
4. Pytest test execution across all 119 active tests;
5. Code quality, formatting, and typing gates (Ruff);
6. Automatic teardown of temporary environments upon completion.

---

## Git Tags & Provenance Anchor

Historical and component tips are tagged:
* `legacy/gnn-mixed-precision-2025`
* `legacy/gnn-push-pull-2025`
* `components/inference-2026`
* `components/adaptation-2026`
* `components/distributed-2026`

## License

Root orchestration and 2026 systems are licensed under Apache-2.0. Historical legacy components retain their original MIT notices. See [LICENSES.md](LICENSES.md).
