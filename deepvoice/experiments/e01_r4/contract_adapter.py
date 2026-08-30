# /// <summary>
# Hash-pinned adapter to the independently audited E00-R2 contract
# /// </summary>

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


DeepvoiceRoot = Path(__file__).resolve().parents[2]
E00ContractPath = DeepvoiceRoot / "experiments" / "e00_r2" / "contract.py"
ExpectedE00ContractSha256 = (
    "b489b136eb80edba8e8a5d6636ae70e273bb677201a6e7da0b3dfb48a2aae4ce"
)


def HashFile(FilePath: Path) -> str:
    Digest = hashlib.sha256()
    with FilePath.open("rb") as FileHandle:
        while True:
            Chunk = FileHandle.read(1024 * 1024)
            if not Chunk:
                break
            Digest.update(Chunk)
    return Digest.hexdigest()


ObservedE00ContractSha256 = HashFile(E00ContractPath)
if ObservedE00ContractSha256 != ExpectedE00ContractSha256:
    raise RuntimeError(
        "E00-R2 contract hash mismatch: "
        f"expected {ExpectedE00ContractSha256}, found {ObservedE00ContractSha256}"
    )

if str(DeepvoiceRoot) not in sys.path:
    sys.path.insert(0, str(DeepvoiceRoot))

from experiments.e00_r2.contract import (  # noqa: E402
    AuditGroupCrossings,
    BootstrapByContentGroup,
    BuildLabelMasks,
    BuildShortcutMetricAudit,
    CalculateCompetitionProxy,
    CalculateHeadMetrics,
    HeadNames,
    HeadWeights,
    LoadManifestPartitions,
    ProjectCrossingRows,
    ValidateNonTestManifestRows,
)


__all__ = (
    "AuditGroupCrossings",
    "BootstrapByContentGroup",
    "BuildLabelMasks",
    "BuildShortcutMetricAudit",
    "CalculateCompetitionProxy",
    "CalculateHeadMetrics",
    "E00ContractPath",
    "ExpectedE00ContractSha256",
    "HashFile",
    "HeadNames",
    "HeadWeights",
    "LoadManifestPartitions",
    "ObservedE00ContractSha256",
    "ProjectCrossingRows",
    "ValidateNonTestManifestRows",
)
