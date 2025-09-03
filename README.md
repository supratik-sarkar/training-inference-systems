
# Push-Pull Parallel GNN with Dynamic Graph Batching

**Goal.** Implement push–pull parallel message passing and dynamic mini-batching for large graphs in PyG.
Evaluate on **OGBN-Products** and **Reddit** datasets.

## Quickstart
```bash
make init
python -m src.push_pull.train --dataset ogbn-products --num-workers 4
```
