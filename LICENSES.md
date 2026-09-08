# Software Licenses

This repository unifies historical graph neural network acceleration research with modern Apple Silicon inference, parameter-efficient adaptation, and distributed training primitives.

| Directory / Subsystem | Era | License | Description |
| :--- | :---: | :--- | :--- |
| `/` (Root Orchestration & Scripts) | 2026 | Apache-2.0 | Root build, verification scripts, and multi-component orchestration |
| `components/inference/` | 2026 | Apache-2.0 | Apple Silicon MLX/PyTorch local LLM inference lab & benchmarking suite |
| `components/adaptation/` | 2026 | Apache-2.0 | Apple Silicon parameter-efficient fine-tuning (LoRA) and rank adaptation |
| `components/distributed/` | 2026 | Apache-2.0 | Distributed training primitives, collective communication, and fault tolerance |
| `legacy/gnn-mixed-precision/` | 2025 | MIT | Historical adaptive mixed-precision graph neural network training |
| `legacy/gnn-push-pull/` | 2025 | MIT | Historical push-pull batching algorithms for graph representations |

Constituent components retain their individual license files in their respective directories.
