
# DeepAdaptiveGNN — Mixed Precision (FP16/BF16)

**Goal.** A reproducible GNN training framework with adaptive depth/width, AMP (FP16/BF16), and MLflow logging.
Benchmarks on **OGBN-ArXiv** and **OGBN-Products**.

## Datasets
- OGBN-ArXiv, OGBN-Products (Open Graph Benchmark).

## Features
- PyTorch Geometric data pipeline.
- Mixed precision with `torch.cuda.amp` and gradient scaling.
- Databricks + local runner (configs in `configs/`).

## Quickstart
```bash
make init
python -m src.train --dataset ogbn-arxiv --precision bf16
```
