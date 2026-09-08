"""Print what each rank physically holds under different placements.

    torchrun --nproc_per_node=4 examples/inspect_placement.py
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from dtp.harness import process_group


def main() -> int:
    with process_group() as ctx:
        rows = 2 * ctx.world_size
        ref = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)

        for label, placement in (("Replicate()", Replicate()),
                                 ("Shard(0)", Shard(0)),
                                 ("Shard(1)", Shard(1))):
            dt = distribute_tensor(ref, ctx.mesh, [placement])
            local = dt.to_local()
            dist.barrier()
            for r in range(ctx.world_size):
                if ctx.rank == r:
                    if r == 0:
                        print(f"\n{label}  global={tuple(dt.shape)}")
                    print(f"  rank {r}: local={tuple(local.shape)} "
                          f"values={local.flatten().tolist()}")
                dist.barrier()

        if ctx.is_leader:
            print("\nEvery placement above reconstructs the same global tensor.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
