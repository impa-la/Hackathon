# /// <summary>
# Resumable exact 16 kHz float32 cache for slow FMA and AIME non-test locators
# /// </summary>

from __future__ import annotations

import hashlib
import functools
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .audio import LoadLocatorWaveform, LoadParquetAudioBytes, ParseLocator
from .records import AudioRecord
from .strict_serialization import JsonBytes, JsonLine


CacheDatasets = frozenset(("fma-small", "aime-open-model-subset"))


def CacheEligible(Record: AudioRecord) -> bool:
    return Record.Dataset in CacheDatasets


def Sha256Bytes(Payload: bytes) -> str:
    return hashlib.sha256(Payload).hexdigest()


def HashFile(FilePath: Path) -> str:
    Digest = hashlib.sha256()
    with FilePath.open("rb") as FileHandle:
        while True:
            Chunk = FileHandle.read(1024 * 1024)
            if not Chunk:
                break
            Digest.update(Chunk)
    return Digest.hexdigest()


def LocatorSha256(Record: AudioRecord) -> str:
    return Sha256Bytes(Record.Locator.encode("utf-8"))


def CacheKey(Record: AudioRecord) -> str:
    Payload = (
        f"{Record.Dataset}\0{Record.SampleId}\0{Record.Locator}"
    ).encode("utf-8")
    return Sha256Bytes(Payload)


def DatasetDirectoryName(Dataset: str) -> str:
    return Dataset.replace("-", "_")


def CachePaths(Record: AudioRecord, CacheRoot: Path) -> tuple[Path, Path]:
    Key = CacheKey(Record)
    DatasetRoot = CacheRoot / DatasetDirectoryName(Record.Dataset)
    return DatasetRoot / f"{Key}.npy", DatasetRoot / f"{Key}.json"


def EstimateCacheStorage(
    Records: Sequence[AudioRecord],
    CacheRoot: Path,
    TargetSampleRate: int,
    SafetyFreeGib: float = 12.0,
) -> dict[str, Any]:
    EligibleRecords = [Record for Record in Records if CacheEligible(Record)]
    EstimatedNpyBytes = sum(
        max(1, math.ceil(Record.DurationSeconds * TargetSampleRate)) * 4 + 128
        for Record in EligibleRecords
    )
    EstimatedMetadataBytes = len(EligibleRecords) * 2048
    EstimatedTotalBytes = EstimatedNpyBytes + EstimatedMetadataBytes
    ExistingBytes = 0
    if CacheRoot.is_dir():
        ExistingBytes = sum(
            FilePath.stat().st_size
            for FilePath in CacheRoot.rglob("*")
            if FilePath.is_file()
        )
    RemainingEstimatedBytes = max(0, EstimatedTotalBytes - ExistingBytes)
    Disk = shutil.disk_usage(CacheRoot.parent if CacheRoot.parent.exists() else CacheRoot.anchor)
    SafetyBytes = int(SafetyFreeGib * 1024**3)
    RequiredFreeBytes = RemainingEstimatedBytes + SafetyBytes
    return {
        "status": "PASS" if Disk.free >= RequiredFreeBytes else "BLOCKED_RESOURCE",
        "eligible_entry_count": len(EligibleRecords),
        "estimated_npy_bytes_from_duration": EstimatedNpyBytes,
        "estimated_metadata_bytes": EstimatedMetadataBytes,
        "estimated_total_cache_bytes": EstimatedTotalBytes,
        "existing_cache_bytes": ExistingBytes,
        "remaining_estimated_bytes": RemainingEstimatedBytes,
        "safety_free_bytes": SafetyBytes,
        "required_current_free_bytes": RequiredFreeBytes,
        "current_free_bytes": Disk.free,
        "projected_free_after_build_bytes": Disk.free - RemainingEstimatedBytes,
        "raw_sources_retained": True,
    }


def SourceAudioSha256(Record: AudioRecord) -> str:
    Parsed = ParseLocator(Record.Locator)
    if Parsed.Kind == "file":
        return HashFile(Parsed.ContainerPath)
    if Parsed.Kind == "parquet":
        ResolvedRow = ResolveAimeManifestRow(Record)["resolved_zero_based_row"]
        return Sha256Bytes(
            LoadParquetAudioBytes(Parsed.ContainerPath, int(ResolvedRow))
        )
    raise ValueError(f"Cache source kind is not allowed for {Record.Dataset}: {Parsed.Kind}")


@functools.lru_cache(maxsize=36)
def LoadParquetIdColumn(ParquetPathText: str):
    import pyarrow.parquet as Parquet

    return Parquet.read_table(Path(ParquetPathText), columns=["id"]).column("id")


def ResolveAimeManifestRow(Record: AudioRecord) -> dict[str, Any]:
    if Record.Dataset != "aime-open-model-subset":
        raise ValueError("AIME row resolution called for another dataset")
    Parsed = ParseLocator(Record.Locator)
    if Parsed.Kind != "parquet" or Parsed.RowIndex is None:
        raise ValueError("AIME record must use a parquet locator")
    DeclaredRow = int(Parsed.RowIndex)
    if DeclaredRow < 1:
        raise RuntimeError(
            f"AIME manifest row must be one-based and positive: {Record.SampleId}"
        )
    ResolvedRow = DeclaredRow - 1
    IdColumn = LoadParquetIdColumn(str(Parsed.ContainerPath))
    if ResolvedRow >= len(IdColumn):
        raise IndexError(
            f"Resolved AIME row {ResolvedRow} is outside {len(IdColumn)} rows"
        )
    ParquetId = str(IdColumn[ResolvedRow].as_py())
    if ParquetId != Record.SampleId:
        raise RuntimeError(
            f"AIME one-based row ID mismatch: manifest {Record.SampleId}, "
            f"parquet {ParquetId}, declared row {DeclaredRow}"
        )
    return {
        "resolver_version": "aime_manifest_one_based_v1",
        "source_locator_index_semantics": "manifest_one_based_resolved_to_parquet_zero_based",
        "declared_manifest_row": DeclaredRow,
        "resolved_zero_based_row": ResolvedRow,
        "asserted_parquet_id": ParquetId,
    }


def AuditAimeLocatorResolution(
    Records: Sequence[AudioRecord],
) -> dict[str, Any]:
    AimeRecords = [
        Record for Record in Records if Record.Dataset == "aime-open-model-subset"
    ]
    DeclaredRows = []
    Shards = set()
    for Record in AimeRecords:
        Parsed = ParseLocator(Record.Locator)
        Resolution = ResolveAimeManifestRow(Record)
        DeclaredRows.append(int(Resolution["declared_manifest_row"]))
        Shards.add(str(Parsed.ContainerPath))
    return {
        "status": "PASS",
        "resolver_version": "aime_manifest_one_based_v1",
        "scope": "non-test AIME records only",
        "checked_record_count": len(AimeRecords),
        "checked_shard_count": len(Shards),
        "declared_row_zero_count": sum(Value == 0 for Value in DeclaredRows),
        "minimum_declared_row": min(DeclaredRows),
        "maximum_declared_row": max(DeclaredRows),
        "parquet_id_mismatch_count": 0,
        "resolution_rule": "resolved_zero_based_row = declared_manifest_row - 1",
    }


def LoadRawRecordWaveform(
    Record: AudioRecord,
    TargetSampleRate: int,
    Diagnostics: dict[str, object] | None = None,
) -> torch.Tensor:
    if Record.Dataset != "aime-open-model-subset":
        return LoadLocatorWaveform(Record.Locator, TargetSampleRate, Diagnostics)
    Parsed = ParseLocator(Record.Locator)
    Resolution = ResolveAimeManifestRow(Record)
    CorrectedLocator = (
        f"parquet://{Parsed.ContainerPath}#row={Resolution['resolved_zero_based_row']}"
    )
    if Diagnostics is not None:
        Diagnostics.update(Resolution)
    return LoadLocatorWaveform(CorrectedLocator, TargetSampleRate, Diagnostics)


def WaveformValueSha256(Waveform: np.ndarray) -> str:
    if Waveform.dtype != np.float32 or Waveform.ndim != 1:
        raise ValueError("Cache waveform hash requires one-dimensional float32")
    return Sha256Bytes(np.ascontiguousarray(Waveform).tobytes())


def WriteAtomic(OutputPath: Path, Payload: bytes) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    TemporaryPath = OutputPath.with_name(OutputPath.name + ".tmp")
    with TemporaryPath.open("wb") as FileHandle:
        FileHandle.write(Payload)
        FileHandle.flush()
        os.fsync(FileHandle.fileno())
    os.replace(TemporaryPath, OutputPath)


def WriteNpyAtomic(OutputPath: Path, Waveform: np.ndarray) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    TemporaryPath = OutputPath.with_name(OutputPath.name + ".tmp")
    with TemporaryPath.open("wb") as FileHandle:
        np.save(FileHandle, Waveform, allow_pickle=False)
        FileHandle.flush()
        os.fsync(FileHandle.fileno())
    os.replace(TemporaryPath, OutputPath)


def LoadStrictJson(InputPath: Path) -> dict[str, Any]:
    return json.loads(
        InputPath.read_text(encoding="utf-8"),
        parse_constant=lambda Value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON constant: {Value}")
        ),
    )


def BuildOrVerifyCacheEntry(
    Record: AudioRecord,
    CacheRoot: Path,
    TargetSampleRate: int,
) -> dict[str, Any]:
    Started = time.perf_counter()
    NpyPath, MetadataPath = CachePaths(Record, CacheRoot)
    BothExist = NpyPath.is_file() and MetadataPath.is_file()
    RecoverNpyOnly = NpyPath.is_file() and not MetadataPath.exists()
    if MetadataPath.exists() and not NpyPath.exists():
        raise RuntimeError(f"Cache metadata exists without array: {MetadataPath}")
    if BothExist:
        Metadata = LoadStrictJson(MetadataPath)
        ExpectedIdentity = {
            "cache_format": "npy-float32-v1",
            "dataset": Record.Dataset,
            "sample_id": Record.SampleId,
            "source_locator": Record.Locator,
            "source_locator_sha256": LocatorSha256(Record),
            "sample_rate_hz": TargetSampleRate,
        }
        for Field, ExpectedValue in ExpectedIdentity.items():
            if Metadata.get(Field) != ExpectedValue:
                raise RuntimeError(
                    f"Cache resume identity mismatch {Field} for {Record.SampleId}"
                )
        if Record.Dataset == "aime-open-model-subset":
            RequiredAimeResolution = ResolveAimeManifestRow(Record)
            for Field, ExpectedValue in RequiredAimeResolution.items():
                if Metadata.get(Field) != ExpectedValue:
                    raise RuntimeError(
                        f"Legacy or mismatched AIME cache sidecar {Field}: {MetadataPath}"
                    )
        Reloaded = np.load(NpyPath, allow_pickle=False)
        if Reloaded.dtype != np.float32 or Reloaded.ndim != 1:
            raise RuntimeError(f"Cache resume dtype/shape mismatch: {NpyPath}")
        if not np.isfinite(Reloaded).all():
            raise RuntimeError(f"Cache resume nonfinite waveform: {NpyPath}")
        if int(Metadata.get("sample_count", -1)) != int(Reloaded.size):
            raise RuntimeError(f"Cache resume sample-count mismatch: {NpyPath}")
        CacheFileSha256 = HashFile(NpyPath)
        WaveformSha256 = WaveformValueSha256(Reloaded)
        if Metadata.get("cache_file_sha256") != CacheFileSha256:
            raise RuntimeError(f"Cache resume file SHA mismatch: {NpyPath}")
        if Metadata.get("waveform_value_sha256") != WaveformSha256:
            raise RuntimeError(f"Cache resume value SHA mismatch: {NpyPath}")
        if Metadata.get("reload_max_absolute_delta") != 0.0:
            raise RuntimeError(f"Prior cache reload delta was not zero: {NpyPath}")
        return {
            **Metadata,
            "status": "VERIFIED_EXISTING_NO_RAW_REDECODE",
            "cache_relative_path": str(NpyPath.relative_to(CacheRoot)).replace("\\", "/"),
            "metadata_relative_path": str(MetadataPath.relative_to(CacheRoot)).replace("\\", "/"),
            "verification_seconds": time.perf_counter() - Started,
        }

    SourceSha256 = SourceAudioSha256(Record)
    RawWaveform = LoadRawRecordWaveform(Record, TargetSampleRate)
    RawArray = np.ascontiguousarray(RawWaveform.cpu().numpy(), dtype=np.float32)
    if RawArray.ndim != 1 or not np.isfinite(RawArray).all():
        raise RuntimeError(f"Raw decoded cache waveform is invalid: {Record.SampleId}")
    if not RecoverNpyOnly:
        WriteNpyAtomic(NpyPath, RawArray)
        Metadata = {
            "cache_format": "npy-float32-v1",
            "dataset": Record.Dataset,
            "sample_id": Record.SampleId,
            "split": Record.Split,
            "source_locator": Record.Locator,
            "source_locator_sha256": LocatorSha256(Record),
            "source_audio_sha256": SourceSha256,
            "sample_rate_hz": TargetSampleRate,
            "sample_count": int(RawArray.size),
            "waveform_value_sha256": WaveformValueSha256(RawArray),
        }
        Status = "BUILT"
    else:
        Metadata = {
            "cache_format": "npy-float32-v1",
            "dataset": Record.Dataset,
            "sample_id": Record.SampleId,
            "split": Record.Split,
            "source_locator": Record.Locator,
            "source_locator_sha256": LocatorSha256(Record),
            "source_audio_sha256": SourceSha256,
            "sample_rate_hz": TargetSampleRate,
            "sample_count": int(RawArray.size),
            "waveform_value_sha256": WaveformValueSha256(RawArray),
        }
        Status = "RECOVERED_PARTIAL"
    if Record.Dataset == "aime-open-model-subset":
        Metadata.update(ResolveAimeManifestRow(Record))
    Reloaded = np.load(NpyPath, allow_pickle=False)
    if Reloaded.dtype != np.float32 or Reloaded.ndim != 1:
        raise RuntimeError(f"Cache dtype/shape mismatch: {NpyPath}")
    if Reloaded.shape != RawArray.shape or not np.isfinite(Reloaded).all():
        raise RuntimeError(f"Cache reload shape/finite mismatch: {NpyPath}")
    MaxAbsoluteDelta = float(np.max(np.abs(Reloaded - RawArray))) if RawArray.size else 0.0
    if MaxAbsoluteDelta != 0.0 or not np.array_equal(Reloaded, RawArray):
        raise RuntimeError(
            f"Cache reload is not exact for {Record.SampleId}: {MaxAbsoluteDelta}"
        )
    CacheFileSha256 = HashFile(NpyPath)
    WaveformSha256 = WaveformValueSha256(Reloaded)
    Metadata.update(
        {
            "cache_file_sha256": CacheFileSha256,
            "cache_file_bytes": NpyPath.stat().st_size,
            "reload_max_absolute_delta": MaxAbsoluteDelta,
            "reload_exact_array_equal": True,
        }
    )
    WriteAtomic(MetadataPath, JsonBytes(Metadata))
    return {
        **Metadata,
        "status": Status,
        "cache_relative_path": str(NpyPath.relative_to(CacheRoot)).replace("\\", "/"),
        "metadata_relative_path": str(MetadataPath.relative_to(CacheRoot)).replace("\\", "/"),
        "verification_seconds": time.perf_counter() - Started,
    }


def BuildExactCache(
    Records: Sequence[AudioRecord],
    CacheRoot: Path,
    TargetSampleRate: int,
) -> dict[str, Any]:
    Started = time.perf_counter()
    EligibleRecords = [Record for Record in Records if CacheEligible(Record)]
    Keys = [CacheKey(Record) for Record in EligibleRecords]
    if len(Keys) != len(set(Keys)):
        raise RuntimeError("Duplicate cache keys detected")
    CacheRoot.mkdir(parents=True, exist_ok=True)
    DiskBefore = shutil.disk_usage(CacheRoot)
    IndexRows = []
    for RecordIndex, Record in enumerate(EligibleRecords):
        IndexRows.append(
            BuildOrVerifyCacheEntry(Record, CacheRoot, TargetSampleRate)
        )
        if (RecordIndex + 1) % 25 == 0:
            Progress = {
                "status": "BUILDING",
                "completed_entries": RecordIndex + 1,
                "expected_entries": len(EligibleRecords),
                "elapsed_seconds": time.perf_counter() - Started,
            }
            WriteAtomic(CacheRoot / "progress.json", JsonBytes(Progress))
    IndexPayload = "".join(JsonLine(Row) + "\n" for Row in IndexRows).encode("utf-8")
    WriteAtomic(CacheRoot / "cache-index.jsonl", IndexPayload)
    IndexSha256 = HashFile(CacheRoot / "cache-index.jsonl")
    BuiltCount = sum(Row["status"] == "BUILT" for Row in IndexRows)
    RecoveredCount = sum(Row["status"] == "RECOVERED_PARTIAL" for Row in IndexRows)
    VerifiedCount = sum(
        Row["status"] == "VERIFIED_EXISTING_NO_RAW_REDECODE" for Row in IndexRows
    )
    CacheBytes = sum(int(Row["cache_file_bytes"]) for Row in IndexRows)
    DiskAfter = shutil.disk_usage(CacheRoot)
    Summary = {
        "status": "PASS",
        "cache_format": "npy-float32-v1",
        "scope": "non-test FMA and AIME only",
        "expected_entries": len(EligibleRecords),
        "completed_entries": len(IndexRows),
        "built_entries": BuiltCount,
        "recovered_partial_entries": RecoveredCount,
        "verified_existing_entries": VerifiedCount,
        "all_reload_max_absolute_delta": 0.0,
        "cache_index_sha256": IndexSha256,
        "cache_npy_bytes": CacheBytes,
        "cache_npy_gib": CacheBytes / 1024**3,
        "disk_free_before_bytes": DiskBefore.free,
        "disk_free_after_bytes": DiskAfter.free,
        "disk_consumed_bytes": DiskBefore.free - DiskAfter.free,
        "build_and_verify_seconds": time.perf_counter() - Started,
        "datasets": {
            Dataset: sum(Record.Dataset == Dataset for Record in EligibleRecords)
            for Dataset in sorted(CacheDatasets)
        },
    }
    WriteAtomic(CacheRoot / "cache-summary.json", JsonBytes(Summary))
    WriteAtomic(
        CacheRoot / "progress.json",
        JsonBytes(
            {
                "status": "COMPLETE",
                "completed_entries": len(IndexRows),
                "expected_entries": len(EligibleRecords),
                "cache_index_sha256": IndexSha256,
            }
        ),
    )
    return Summary


def VerifyCompleteCache(
    Records: Sequence[AudioRecord],
    CacheRoot: Path,
) -> dict[str, Any]:
    EligibleRecords = [Record for Record in Records if CacheEligible(Record)]
    SummaryPath = CacheRoot / "cache-summary.json"
    IndexPath = CacheRoot / "cache-index.jsonl"
    if not SummaryPath.is_file() or not IndexPath.is_file():
        return {"status": "BLOCKED", "reason": "cache summary/index missing"}
    Summary = LoadStrictJson(SummaryPath)
    Checks = {
        "summary_pass": Summary.get("status") == "PASS",
        "entry_count": Summary.get("completed_entries") == len(EligibleRecords),
        "expected_count": Summary.get("expected_entries") == len(EligibleRecords),
        "reload_delta_zero": Summary.get("all_reload_max_absolute_delta") == 0.0,
        "index_sha256": Summary.get("cache_index_sha256") == HashFile(IndexPath),
        "every_entry_present": all(
            all(PathValue.is_file() for PathValue in CachePaths(Record, CacheRoot))
            for Record in EligibleRecords
        ),
    }
    return {
        "status": "PASS" if all(Checks.values()) else "BLOCKED",
        "checks": Checks,
        "summary": Summary,
    }


def LoadCachedOrRawWaveform(
    Record: AudioRecord,
    CacheRoot: Path,
    TargetSampleRate: int,
    Diagnostics: dict[str, object] | None = None,
) -> torch.Tensor:
    if not CacheEligible(Record):
        return LoadLocatorWaveform(Record.Locator, TargetSampleRate, Diagnostics)
    NpyPath, MetadataPath = CachePaths(Record, CacheRoot)
    if not NpyPath.is_file() or not MetadataPath.is_file():
        raise FileNotFoundError(f"Required exact cache entry is missing: {NpyPath}")
    Array = np.load(NpyPath, mmap_mode="c", allow_pickle=False)
    if Array.dtype != np.float32 or Array.ndim != 1:
        raise RuntimeError(f"Cached waveform dtype/shape mismatch: {NpyPath}")
    if Diagnostics is not None:
        Diagnostics["decoder_backend"] = "exact_npy_float32_mmap"
    return torch.from_numpy(Array)
