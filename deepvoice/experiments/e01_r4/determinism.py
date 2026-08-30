# /// <summary>
# Explicit CPU thread invariants for byte-exact loader augmentation on Windows
# /// </summary>

from __future__ import annotations

from typing import Any

import torch


def ConfigureParentCpuThreads(Config: dict[str, Any]) -> dict[str, Any]:
    IntraopThreads = int(Config["cpu_parent_intraop_threads"])
    InteropThreads = int(Config["cpu_parent_interop_threads"])
    if IntraopThreads != 1 or InteropThreads != 1:
        raise RuntimeError("E01-R4 deterministic CPU thread counts must both equal one")
    torch.set_num_threads(IntraopThreads)
    if torch.get_num_interop_threads() != InteropThreads:
        try:
            torch.set_num_interop_threads(InteropThreads)
        except RuntimeError as Error:
            raise RuntimeError(
                "E01-R4 interop threads were initialized before the deterministic "
                "one-thread invariant"
            ) from Error
    Evidence = {
        "parent_intraop_threads": torch.get_num_threads(),
        "parent_interop_threads": torch.get_num_interop_threads(),
        "worker_intraop_threads": int(Config["cpu_worker_intraop_threads"]),
    }
    if Evidence != {
        "parent_intraop_threads": 1,
        "parent_interop_threads": 1,
        "worker_intraop_threads": 1,
    }:
        raise RuntimeError(f"CPU thread invariant failed: {Evidence}")
    return Evidence


def PinDataLoaderWorkerCpuThreads(WorkerId: int) -> None:
    del WorkerId
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("DataLoader worker intraop thread pin failed")
