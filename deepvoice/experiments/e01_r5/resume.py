# /// <summary>
# Strict atomic identity, status, progress and resume primitives for E01-R5
# /// </summary>

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from .strict_serialization import AssertFinitePayload, JsonBytes


ResumeSchemaVersion = "e01-r5-epoch-resume-v1"
CompleteSeedSchemaVersion = "e01-r5-complete-seed-v1"


def HashFile(FilePath: Path) -> str:
    Digest = hashlib.sha256()
    with FilePath.open("rb") as FileHandle:
        while True:
            Chunk = FileHandle.read(1024 * 1024)
            if not Chunk:
                break
            Digest.update(Chunk)
    return Digest.hexdigest()


def AtomicWriteBytes(OutputPath: Path, Payload: bytes) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    TemporaryPath = OutputPath.with_name(f"{OutputPath.name}.tmp-{os.getpid()}")
    with TemporaryPath.open("wb") as FileHandle:
        FileHandle.write(Payload)
        FileHandle.flush()
        os.fsync(FileHandle.fileno())
    os.replace(TemporaryPath, OutputPath)


def AtomicWriteJson(OutputPath: Path, Payload: Any) -> None:
    AtomicWriteBytes(OutputPath, JsonBytes(Payload))


def StrictLoadJson(InputPath: Path) -> Any:
    return json.loads(
        InputPath.read_text(encoding="utf-8"),
        parse_constant=lambda Value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON constant: {Value}")
        ),
    )


def AtomicTorchSave(OutputPath: Path, Payload: dict[str, Any]) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    TemporaryPath = OutputPath.with_name(f"{OutputPath.name}.tmp-{os.getpid()}")
    torch.save(Payload, TemporaryPath)
    with TemporaryPath.open("rb+") as FileHandle:
        FileHandle.flush()
        os.fsync(FileHandle.fileno())
    os.replace(TemporaryPath, OutputPath)


def BuildCodeInventory(SourceRoot: Path) -> list[dict[str, Any]]:
    Rows = []
    for FilePath in sorted(SourceRoot.iterdir(), key=lambda Value: Value.name):
        if not FilePath.is_file() or FilePath.suffix == ".pyc":
            continue
        Rows.append(
            {
                "relative_path": FilePath.name,
                "bytes": FilePath.stat().st_size,
                "sha256": HashFile(FilePath),
            }
        )
    return Rows


def InventoryDigest(Rows: list[dict[str, Any]]) -> str:
    Digest = hashlib.sha256()
    for Row in Rows:
        Digest.update(
            f"{Row['relative_path']}\0{Row['bytes']}\0{Row['sha256']}\n".encode(
                "utf-8"
            )
        )
    return Digest.hexdigest()


def BuildRunIdentity(
    DeepvoiceRoot: Path,
    SourceRoot: Path,
    ConfigPath: Path,
    Config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ManifestPath = DeepvoiceRoot / Config["manifest_relative_path"]
    CacheRoot = DeepvoiceRoot / Config["cache_relative_path"]
    CacheSummaryPath = CacheRoot / "cache-summary.json"
    CacheIndexPath = CacheRoot / "cache-index.jsonl"
    R4AuditPath = DeepvoiceRoot / Config["e01_r4_audit_relative_path"]
    RequiredPaths = (ManifestPath, CacheSummaryPath, CacheIndexPath, R4AuditPath)
    MissingPaths = [str(PathValue) for PathValue in RequiredPaths if not PathValue.is_file()]
    if MissingPaths:
        raise FileNotFoundError(f"R5 identity inputs are missing: {MissingPaths}")
    ManifestSha256 = HashFile(ManifestPath)
    if ManifestSha256 != Config["manifest_sha256"]:
        raise RuntimeError("R5 manifest SHA does not match the fixed config contract")
    R4AuditSha256 = HashFile(R4AuditPath)
    if R4AuditSha256 != Config["e01_r4_audit_sha256"]:
        raise RuntimeError("R5 R4-audit SHA does not match the fixed config contract")
    R4AuditFirstLine = R4AuditPath.read_text(encoding="utf-8").splitlines()[0]
    if R4AuditFirstLine != Config["e01_r4_audit_required_first_line"]:
        raise RuntimeError("R5 requires a PASS first line from the R4 audit")
    CacheSummary = StrictLoadJson(CacheSummaryPath)
    if CacheSummary.get("status") != "PASS":
        raise RuntimeError("R5 exact cache summary is not PASS")
    CacheIndexSha256 = HashFile(CacheIndexPath)
    if CacheSummary.get("cache_index_sha256") != CacheIndexSha256:
        raise RuntimeError("R5 cache index SHA does not match cache summary")
    CodeInventory = BuildCodeInventory(SourceRoot)
    Identity = {
        "identity_schema": "e01-r5-run-identity-v1",
        "experiment_id": "E01",
        "revision": "R5",
        "config_sha256": HashFile(ConfigPath),
        "code_inventory_sha256": InventoryDigest(CodeInventory),
        "manifest_sha256": ManifestSha256,
        "cache_summary_sha256": HashFile(CacheSummaryPath),
        "cache_index_sha256": CacheIndexSha256,
        "cache_completed_entries": int(CacheSummary["completed_entries"]),
        "e00_r2_contract_sha256": Config["e00_r2_contract_sha256"],
        "e01_r4_audit_sha256": R4AuditSha256,
    }
    AssertFinitePayload(Identity)
    return Identity, CodeInventory


def ValidateRunIdentity(Expected: dict[str, Any], Observed: dict[str, Any]) -> None:
    if not isinstance(Observed, dict):
        raise RuntimeError("Resume identity is not an object")
    ExpectedKeys = set(Expected)
    ObservedKeys = set(Observed)
    if ExpectedKeys != ObservedKeys:
        raise RuntimeError(
            "Resume identity keys mismatch: "
            f"missing={sorted(ExpectedKeys - ObservedKeys)}, "
            f"unexpected={sorted(ObservedKeys - ExpectedKeys)}"
        )
    Mismatches = [
        Key for Key in sorted(ExpectedKeys) if Expected[Key] != Observed[Key]
    ]
    if Mismatches:
        raise RuntimeError(f"Resume identity hash mismatch: {Mismatches}")


def ProgressMath(
    CompletedUnits: int,
    TotalUnits: int,
    IntervalCompletedUnits: int,
    IntervalElapsedSeconds: float,
    RunElapsedSeconds: float,
) -> dict[str, float]:
    if TotalUnits <= 0 or not 0 <= CompletedUnits <= TotalUnits:
        raise ValueError("Invalid overall progress units")
    if IntervalCompletedUnits <= 0 or IntervalElapsedSeconds <= 0.0:
        raise ValueError("Progress rate requires positive interval units and seconds")
    Rate = IntervalCompletedUnits / IntervalElapsedSeconds
    RemainingUnits = TotalUnits - CompletedUnits
    EtaSeconds = RemainingUnits / Rate
    Payload = {
        "overall_percent": CompletedUnits * 100.0 / TotalUnits,
        "units_per_second": Rate,
        "elapsed_seconds": RunElapsedSeconds,
        "eta_seconds": EtaSeconds,
    }
    AssertFinitePayload(Payload)
    return Payload


def FormatDuration(Seconds: float) -> str:
    if not math.isfinite(Seconds) or Seconds < 0.0:
        raise ValueError("Duration must be finite and nonnegative")
    Rounded = int(round(Seconds))
    Hours, Remainder = divmod(Rounded, 3600)
    Minutes, SecondsPart = divmod(Remainder, 60)
    return f"{Hours:02d}:{Minutes:02d}:{SecondsPart:02d}"


def CsvBytes(Rows: list[dict[str, Any]]) -> bytes:
    if not Rows:
        raise ValueError("Cannot serialize an empty CSV")
    AssertFinitePayload(Rows)
    import io

    Buffer = io.StringIO(newline="")
    Writer = csv.DictWriter(Buffer, fieldnames=list(Rows[0]))
    Writer.writeheader()
    Writer.writerows(Rows)
    return Buffer.getvalue().encode("utf-8")


def UtcNow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
