"""Process-group setup for local CPU runs.

Everything here uses the Gloo backend across ``torchrun`` worker processes on
one machine. That is a *semantics* laboratory, not a performance one: the goal
is to prove that a sharded computation agrees with its unsharded reference,
which is a property of the algorithm and not of the hardware.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


@dataclass(frozen=True, slots=True)
class DistContext:
    rank: int
    world_size: int
    mesh: DeviceMesh
    device: str = "cpu"

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


@contextmanager
def process_group(mesh_shape: tuple[int, ...] | None = None
                  ) -> Iterator[DistContext]:
    """Initialise Gloo, build a DeviceMesh, and tear both down cleanly."""
    if not is_distributed():
        raise RuntimeError(
            "not launched under torchrun; run scripts/run_experiments.sh"
        )
    dist.init_process_group(backend="gloo")
    try:
        world = dist.get_world_size()
        shape = mesh_shape or (world,)
        if int(torch.tensor(shape).prod()) != world:
            raise ValueError(f"mesh {shape} does not cover world size {world}")
        mesh = init_device_mesh("cpu", shape)
        yield DistContext(rank=dist.get_rank(), world_size=world, mesh=mesh)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def distributed_main(fn: Callable[[DistContext], int]) -> int:
    with process_group() as ctx:
        return fn(ctx)


def seeded(seed: int = 1234) -> torch.Generator:
    """A generator seeded identically on every rank.

    Every experiment builds its reference tensors from this, so all ranks hold
    the same reference and the equivalence check is meaningful.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


__all__ = ["DistContext", "distributed_main", "is_distributed",
           "process_group", "seeded"]
