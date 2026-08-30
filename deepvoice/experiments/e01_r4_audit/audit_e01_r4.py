# /// <summary>
# Independent read-only auditor for DeepVoice E01-R4 cache/benchmark readiness.
# /// </summary>

from __future__ import annotations

import csv
import gzip
import hashlib
import inspect
import json
import math
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ExpectedManifestSha256 = (
    "2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6"
)
ExpectedManifestRows = 137328
ExpectedCacheEntries = 5651
ExpectedCacheIndexSha256 = (
    "ff27584b19039226557dc12dc62a91f6cf8f723e711737406ae77341ab9ef13a"
)
ExpectedCacheNpyBytes = 9557290108
R4JsonReports = (
    "e01-r4-aime-row-resolution.json",
    "e01-r4-batch-autotune.json",
    "e01-r4-cache-build.json",
    "e01-r4-cache-integrity.json",
    "e01-r4-cache-storage-gate.json",
    "e01-r4-cached-gpu-pilot.json",
    "e01-r4-preflight.json",
    "e01-r4-run-manifest.json",
    "e01-r4-runtime-projection.json",
    "e01-r4-tiny-gpu-smoke.json",
    "e01-r4-unit-test-results.json",
    "e01-r4-windows-workers.json",
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


def Sha256Bytes(Payload: bytes) -> str:
    return hashlib.sha256(Payload).hexdigest()


def RejectConstant(Value: str) -> None:
    raise ValueError(f"non-standard JSON constant {Value}")


def AssertFiniteRecursive(Value: Any, PathText: str = "$") -> None:
    if Value is None or isinstance(Value, (bool, str, int)):
        return
    if isinstance(Value, float):
        if not math.isfinite(Value):
            raise ValueError(f"nonfinite scalar at {PathText}: {Value!r}")
        return
    if isinstance(Value, Mapping):
        for Key, Nested in Value.items():
            AssertFiniteRecursive(Nested, f"{PathText}.{Key}")
        return
    if isinstance(Value, Sequence) and not isinstance(Value, (bytes, bytearray)):
        for Index, Nested in enumerate(Value):
            AssertFiniteRecursive(Nested, f"{PathText}[{Index}]")
        return
    raise TypeError(f"unsupported JSON value at {PathText}: {type(Value).__name__}")


def LoadStrictJson(FilePath: Path) -> Any:
    Value = json.loads(
        FilePath.read_text(encoding="utf-8"),
        parse_constant=RejectConstant,
    )
    AssertFiniteRecursive(Value)
    return Value


def LoadCsv(FilePath: Path) -> list[dict[str, str]]:
    with FilePath.open("r", encoding="utf-8-sig", newline="") as FileHandle:
        return list(csv.DictReader(FileHandle))


def AddCheck(
    Checks: list[dict[str, Any]],
    Name: str,
    Errors: Sequence[str],
    Evidence: Any,
) -> None:
    Checks.append(
        {
            "check": Name,
            "status": "PASS" if not Errors else "BLOCKED",
            "errors": list(Errors),
            "evidence": Evidence,
        }
    )


def ReadManifest(
    ManifestPath: Path,
) -> tuple[int, list[dict[str, str]], list[dict[str, str]], list[str]]:
    RowCount = 0
    TrainingRows = []
    ValidationRows = []
    GroupSplits: dict[str, set[str]] = defaultdict(set)
    with gzip.open(ManifestPath, "rt", encoding="utf-8-sig", newline="") as FileHandle:
        Reader = csv.DictReader(FileHandle)
        for Row in Reader:
            RowCount += 1
            Split = Row["recommended_content_split"]
            GroupSplits[Row["content_group_key"]].add(Split)
            if Split == "test":
                continue
            if Split == "train":
                TrainingRows.append(Row)
            elif Split == "validation":
                ValidationRows.append(Row)
            else:
                raise ValueError(f"unexpected manifest split {Split}")
    Crossings = sorted(
        Group for Group, Splits in GroupSplits.items() if len(Splits) > 1
    )
    return RowCount, TrainingRows, ValidationRows, Crossings


def AuditManifest(
    ManifestPath: Path,
    RunRecord: dict[str, Any],
    RecordsModule: Any,
) -> tuple[
    dict[str, Any],
    list[str],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    Errors = []
    RowCount, TrainingRows, ValidationRows, Crossings = ReadManifest(ManifestPath)
    ManifestHash = HashFile(ManifestPath)
    if ManifestHash != ExpectedManifestSha256:
        Errors.append(f"manifest SHA mismatch {ManifestHash}")
    if RowCount != ExpectedManifestRows:
        Errors.append(f"manifest row count {RowCount}")
    if Crossings:
        Errors.append(f"content-group crossing count {len(Crossings)}")
    Recorded = RunRecord["manifest_integrity"]
    ExpectedRecorded = {
        "manifest_sha256": ManifestHash,
        "manifest_total_row_count": RowCount,
        "crossing_group_count": 0,
        "train_row_count": len(TrainingRows),
        "validation_row_count": len(ValidationRows),
        "test_field_contract": {
            "allowed_fields": ["content_group_key", "recommended_content_split"],
            "retained_forbidden_fields": 0,
            "test_statistics": 0,
        },
    }
    if Recorded != ExpectedRecorded:
        Errors.append("R4 run manifest integrity block differs from independent audit")
    TestUsage = RunRecord["test_usage_contract"]
    if TestUsage != {
        "allowed_fields": ["content_group_key", "recommended_content_split"],
        "test_metrics": 0,
        "test_predictions": 0,
        "test_statistics": 0,
    }:
        Errors.append("R4 run manifest test-usage contract differs")
    Source = inspect.getsource(RecordsModule.LoadE01Records)
    if "BuildLabelMasks(NonTestRows)" not in Source:
        Errors.append("record loader does not restrict labels/masks to non-test rows")
    if "ProjectCrossingRows(NonTestRows) + TestCrossingRows" not in Source:
        Errors.append("record loader does not isolate test rows to crossing projection")
    return (
        {
            "manifest_sha256": ManifestHash,
            "row_count": RowCount,
            "train_rows": len(TrainingRows),
            "validation_rows": len(ValidationRows),
            "crossing_group_count": len(Crossings),
            "test_handling": (
                "group/split crossing projection only; test rows discarded immediately, "
                "with no label/mask/metadata/performance statistic retained"
            ),
        },
        Errors,
        TrainingRows,
        ValidationRows,
    )


def AuditInventory(
    SourceRoot: Path,
    InventoryPath: Path,
    RunRecord: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    Rows = LoadCsv(InventoryPath)
    CurrentNames = sorted(
        FilePath.name
        for FilePath in SourceRoot.glob("*")
        if FilePath.is_file() and FilePath.suffix != ".pyc"
    )
    RecordedNames = sorted(Row["relative_path"] for Row in Rows)
    if CurrentNames != RecordedNames:
        Errors.append("R4 source file set differs from inventory")
    DigestPayload = bytearray()
    Results = []
    for Row in Rows:
        FilePath = SourceRoot / Row["relative_path"]
        ActualBytes = FilePath.stat().st_size
        ActualHash = HashFile(FilePath)
        Match = ActualBytes == int(Row["bytes"]) and ActualHash == Row["sha256"]
        if not Match:
            Errors.append(f"{Row['relative_path']} differs from R4 inventory")
        Results.append(
            {
                "relative_path": Row["relative_path"],
                "bytes": ActualBytes,
                "sha256": ActualHash,
                "match": Match,
            }
        )
        DigestPayload.extend(
            f"{Row['relative_path']}\0{Row['bytes']}\0{Row['sha256']}\n".encode(
                "utf-8"
            )
        )
    Digest = hashlib.sha256(DigestPayload).hexdigest()
    RecordedDigest = RunRecord["versions"]["e01_r4_code_inventory_sha256"]
    if Digest != RecordedDigest:
        Errors.append("R4 inventory digest differs from run manifest")
    return {
        "file_count": len(Results),
        "inventory_sha256": Digest,
        "recorded_inventory_sha256": RecordedDigest,
        "files": Results,
    }, Errors


def AuditStrictJson(
    ReportsRoot: Path,
    ArtifactPath: Path,
    CacheRoot: Path,
    SerializationModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    OutputFiles = [ReportsRoot / Name for Name in R4JsonReports] + [ArtifactPath]
    Results = []
    for FilePath in OutputFiles:
        try:
            LoadStrictJson(FilePath)
            IsStrict = True
            ErrorText = None
        except (json.JSONDecodeError, ValueError, TypeError) as Error:
            IsStrict = False
            ErrorText = str(Error)
            Errors.append(f"{FilePath.name} strict JSON failure: {ErrorText}")
        Results.append(
            {
                "file": FilePath.name,
                "sha256": HashFile(FilePath),
                "strict_recursive_finite": IsStrict,
                "error": ErrorText,
            }
        )
    for CacheControlName in ("cache-summary.json", "progress.json"):
        try:
            LoadStrictJson(CacheRoot / CacheControlName)
        except (json.JSONDecodeError, ValueError, TypeError) as Error:
            Errors.append(f"{CacheControlName} strict JSON failure: {Error}")
    NestedNanRejected = False
    NestedPathReported = False
    try:
        SerializationModule.JsonBytes({"outer": [{"inner": float("nan")}]})
    except SerializationModule.NonFinitePayloadError as Error:
        NestedNanRejected = True
        NestedPathReported = "$.outer[0].inner" in str(Error)
    if not NestedNanRejected or not NestedPathReported:
        Errors.append("recursive strict serializer did not reject nested NaN with path")
    JsonBytesSource = inspect.getsource(SerializationModule.JsonBytes)
    JsonLineSource = inspect.getsource(SerializationModule.JsonLine)
    if "allow_nan=False" not in JsonBytesSource or "allow_nan=False" not in JsonLineSource:
        Errors.append("strict JSON writers do not both force allow_nan=False")
    return {
        "strict_output_count": sum(
            Row["strict_recursive_finite"] for Row in Results
        ),
        "output_count": len(Results),
        "outputs": Results,
        "nested_nan_rejected": NestedNanRejected,
        "nested_error_path_reported": NestedPathReported,
        "json_bytes_allow_nan_false": "allow_nan=False" in JsonBytesSource,
        "json_line_allow_nan_false": "allow_nan=False" in JsonLineSource,
    }, Errors


def AuditTestsAndNumerics(
    Config: dict[str, Any],
    RecordedTests: dict[str, Any],
    Pilot: dict[str, Any],
    RunTestsModule: Any,
    NumericalModule: Any,
    BenchmarkModule: Any,
    TrainModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    ActualTests = RunTestsModule.RunAllTests(Config)
    if ActualTests.get("status") != "PASS" or ActualTests.get("check_count") != 15:
        Errors.append(f"fresh unit/adversarial tests failed: {ActualTests}")
    if RecordedTests.get("status") != "PASS" or RecordedTests.get("check_count") != 15:
        Errors.append("recorded R4 test report is not 15/15 PASS")
    RequiredTestFunctions = (
        "CheckMaskedNanLogitAdversary",
        "CheckStrictNestedJsonRejectsNan",
        "CheckSkippedOptimizerStepRejected",
        "CheckFp32TrainingMode",
        "CheckWindowsWorkerTaggedDeterminism",
    )
    RunAllSource = inspect.getsource(RunTestsModule.RunAllTests)
    MissingTests = [Name for Name in RequiredTestFunctions if Name not in RunAllSource]
    if MissingTests:
        Errors.append(f"required adversarial tests absent from RunAllTests: {MissingTests}")

    if not torch.cuda.is_available():
        Errors.append("CUDA unavailable for fresh guarded FP32 smoke")
        FreshSmoke = {"status": "BLOCKED", "reason": "CUDA unavailable"}
    else:
        FreshSmoke = RunTestsModule.RunTinySmoke(Config, "cuda")
        if FreshSmoke.get("status") != "PASS":
            Errors.append(f"fresh guarded CUDA smoke failed: {FreshSmoke}")
        if FreshSmoke.get("optimizer_step_after") != FreshSmoke.get(
            "optimizer_step_before", -1
        ) + 1:
            Errors.append("fresh CUDA smoke did not prove exactly one optimizer step")
        if not math.isfinite(float(FreshSmoke.get("loss", float("nan")))):
            Errors.append("fresh CUDA smoke loss is nonfinite")
        if FreshSmoke.get("autocast_enabled") is not False or FreshSmoke.get(
            "grad_scaler_enabled"
        ) is not False:
            Errors.append("fresh CUDA smoke did not remain FP32-only")

    GuardSource = inspect.getsource(NumericalModule.GuardedFp32OptimizationStep)
    RequiredGuardMarkers = (
        "RequireFiniteTensor(Logits",
        "RequireFiniteTensor(Loss",
        "RequireModelParametersFinite(Model",
        "Loss.backward()",
        "RequireGradientsFinite(Model",
        "clip_grad_norm_",
        "Optimizer.step()",
        "StepAfter != StepBefore + 1",
        '"after_step"',
    )
    MissingGuardMarkers = [
        Marker for Marker in RequiredGuardMarkers if Marker not in GuardSource
    ]
    if MissingGuardMarkers:
        Errors.append(f"guarded step is missing checks: {MissingGuardMarkers}")
    PilotSource = inspect.getsource(BenchmarkModule.BenchmarkCachedEndToEndPilot)
    FullSource = inspect.getsource(TrainModule.RunSeed)
    if "GuardedFp32OptimizationStep(" not in PilotSource:
        Errors.append("cached GPU pilot does not call the guarded FP32 step")
    if "GuardedFp32OptimizationStep(" not in FullSource:
        Errors.append("future full RunSeed does not call the guarded FP32 step")
    if Pilot.get("status") != "PASS" or Pilot.get("batch_count") != 32:
        Errors.append("recorded cached GPU pilot is not 32-batch PASS")
    if Pilot.get("guarded_batch_count") != Pilot.get("batch_count"):
        Errors.append("not every cached pilot batch is guarded")
    if Pilot.get("optimizer_skip_count") != 0:
        Errors.append("cached pilot reports optimizer skips")
    PilotFiniteFields = (
        "final_loss",
        "maximum_fp32_gradient_norm",
        "end_to_end_samples_per_second",
        "peak_allocated_gib",
    )
    if any(not math.isfinite(float(Pilot[Field])) for Field in PilotFiniteFields):
        Errors.append("cached pilot contains a nonfinite numerical field")
    if Pilot.get("precision_mode") != "fp32_guarded" or Pilot.get(
        "autocast_enabled"
    ) is not False or Pilot.get("grad_scaler_enabled") is not False:
        Errors.append("cached pilot precision evidence is not FP32-only")
    return {
        "fresh_test_status": ActualTests.get("status"),
        "fresh_test_count": ActualTests.get("check_count"),
        "recorded_test_count": RecordedTests.get("check_count"),
        "required_adversarial_tests_in_suite": not MissingTests,
        "fresh_cuda_smoke": FreshSmoke,
        "guard_markers_present": not MissingGuardMarkers,
        "pilot": {
            "status": Pilot.get("status"),
            "batch_count": Pilot.get("batch_count"),
            "guarded_batch_count": Pilot.get("guarded_batch_count"),
            "optimizer_skip_count": Pilot.get("optimizer_skip_count"),
            "final_loss": Pilot.get("final_loss"),
            "maximum_fp32_gradient_norm": Pilot.get(
                "maximum_fp32_gradient_norm"
            ),
            "precision_mode": Pilot.get("precision_mode"),
            "autocast_enabled": Pilot.get("autocast_enabled"),
            "grad_scaler_enabled": Pilot.get("grad_scaler_enabled"),
        },
    }, Errors


def CacheRecordKey(Dataset: str, SampleId: str, Locator: str) -> tuple[str, str, str]:
    return Dataset, SampleId, Locator


def AuditCache(
    CacheRoot: Path,
    QuarantineRoot: Path,
    EligibleRecords: Sequence[Any],
    ManifestRows: Sequence[dict[str, str]],
    CacheSummaryReport: dict[str, Any],
    CacheBuildReport: dict[str, Any],
    CacheModule: Any,
) -> tuple[dict[str, Any], list[str], dict[tuple[str, str, str], dict[str, Any]]]:
    Errors = []
    IndexPath = CacheRoot / "cache-index.jsonl"
    IndexHash = HashFile(IndexPath)
    if IndexHash != ExpectedCacheIndexSha256:
        Errors.append(f"cache index SHA mismatch {IndexHash}")
    IndexRows = []
    with IndexPath.open("r", encoding="utf-8") as FileHandle:
        for LineNumber, Line in enumerate(FileHandle, 1):
            try:
                Row = json.loads(Line, parse_constant=RejectConstant)
                AssertFiniteRecursive(Row, f"index_line_{LineNumber}")
            except (json.JSONDecodeError, ValueError, TypeError) as Error:
                Errors.append(f"cache index line {LineNumber} invalid: {Error}")
                continue
            IndexRows.append(Row)
    if len(IndexRows) != ExpectedCacheEntries:
        Errors.append(f"cache index entries {len(IndexRows)}")
    IndexByKey = {
        CacheRecordKey(Row["dataset"], Row["sample_id"], Row["source_locator"]): Row
        for Row in IndexRows
    }
    if len(IndexByKey) != len(IndexRows):
        Errors.append("cache index has duplicate dataset/sample/locator keys")
    EligibleKeys = {
        CacheRecordKey(Record.Dataset, Record.SampleId, Record.Locator)
        for Record in EligibleRecords
    }
    if set(IndexByKey) != EligibleKeys:
        Errors.append("cache index keys differ from non-test eligible records")

    ManifestByKey = {
        CacheRecordKey(Row["dataset"], Row["sample_id"], Row["source_locator"]): Row
        for Row in ManifestRows
        if Row["dataset"] in ("fma-small", "aime-open-model-subset")
    }
    DatasetCounts = Counter()
    DtypeCounts = Counter()
    NpyBytes = 0
    HeaderErrors = 0
    SidecarErrors = 0
    ManifestHashMismatchCount = 0
    QuarantineReferenceCount = 0
    ActiveRootResolved = CacheRoot.resolve()
    for RowIndex, Row in enumerate(IndexRows):
        DatasetCounts[Row["dataset"]] += 1
        RelativeNpy = Path(Row["cache_relative_path"])
        RelativeMetadata = Path(Row["metadata_relative_path"])
        NpyPath = (CacheRoot / RelativeNpy).resolve()
        MetadataPath = (CacheRoot / RelativeMetadata).resolve()
        try:
            NpyPath.relative_to(ActiveRootResolved)
            MetadataPath.relative_to(ActiveRootResolved)
        except ValueError:
            HeaderErrors += 1
            continue
        if "quarantine" in str(NpyPath).casefold() or "quarantine" in str(
            MetadataPath
        ).casefold():
            QuarantineReferenceCount += 1
        if not NpyPath.is_file() or not MetadataPath.is_file():
            HeaderErrors += 1
            continue
        NpyBytes += NpyPath.stat().st_size
        try:
            Array = np.load(NpyPath, mmap_mode="r", allow_pickle=False)
            DtypeCounts[str(Array.dtype)] += 1
            if Array.dtype != np.float32 or Array.ndim != 1:
                HeaderErrors += 1
            if int(Array.size) != int(Row["sample_count"]):
                HeaderErrors += 1
            if NpyPath.stat().st_size != int(Row["cache_file_bytes"]):
                HeaderErrors += 1
            del Array
        except Exception:
            HeaderErrors += 1
        try:
            Metadata = LoadStrictJson(MetadataPath)
            for Field in (
                "dataset",
                "sample_id",
                "source_locator",
                "source_locator_sha256",
                "source_audio_sha256",
                "sample_rate_hz",
                "sample_count",
                "waveform_value_sha256",
                "cache_file_sha256",
                "cache_file_bytes",
            ):
                if Metadata.get(Field) != Row.get(Field):
                    SidecarErrors += 1
                    break
        except (json.JSONDecodeError, ValueError, TypeError):
            SidecarErrors += 1
        Key = CacheRecordKey(Row["dataset"], Row["sample_id"], Row["source_locator"])
        ManifestRow = ManifestByKey.get(Key)
        if ManifestRow is None or Row["source_audio_sha256"] != ManifestRow[
            "file_sha256"
        ]:
            ManifestHashMismatchCount += 1
    if HeaderErrors:
        Errors.append(f"cache NPY header/path errors {HeaderErrors}")
    if SidecarErrors:
        Errors.append(f"cache sidecar consistency errors {SidecarErrors}")
    if ManifestHashMismatchCount:
        Errors.append(
            f"cache source_audio_sha256/manifest file_sha256 mismatches {ManifestHashMismatchCount}"
        )
    if QuarantineReferenceCount:
        Errors.append(f"active index references quarantine {QuarantineReferenceCount} times")
    NpyFiles = [
        FilePath
        for DatasetRoot in (
            CacheRoot / "fma_small",
            CacheRoot / "aime_open_model_subset",
        )
        for FilePath in DatasetRoot.glob("*.npy")
    ]
    SidecarFiles = [
        FilePath
        for DatasetRoot in (
            CacheRoot / "fma_small",
            CacheRoot / "aime_open_model_subset",
        )
        for FilePath in DatasetRoot.glob("*.json")
    ]
    if len(NpyFiles) != ExpectedCacheEntries or len(SidecarFiles) != ExpectedCacheEntries:
        Errors.append(
            f"active cache file counts npy={len(NpyFiles)} json={len(SidecarFiles)}"
        )
    if NpyBytes != ExpectedCacheNpyBytes:
        Errors.append(f"cache NPY bytes {NpyBytes}")
    if DatasetCounts != Counter(
        {"fma-small": 4642, "aime-open-model-subset": 1009}
    ):
        Errors.append(f"cache dataset counts differ {dict(DatasetCounts)}")
    if DtypeCounts != Counter({"float32": ExpectedCacheEntries}):
        Errors.append(f"cache dtype counts differ {dict(DtypeCounts)}")
    Summary = CacheSummaryReport["summary"]
    for Field, Expected in (
        ("completed_entries", ExpectedCacheEntries),
        ("expected_entries", ExpectedCacheEntries),
        ("cache_index_sha256", IndexHash),
        ("cache_npy_bytes", NpyBytes),
        ("all_reload_max_absolute_delta", 0.0),
    ):
        if Summary.get(Field) != Expected:
            Errors.append(f"cache integrity summary {Field} differs")
    if CacheSummaryReport.get("status") != "PASS" or not all(
        CacheSummaryReport.get("checks", {}).values()
    ):
        Errors.append("cache integrity report does not pass every check")
    if CacheBuildReport.get("action") != "REUSED_COMPLETE_NO_RAW_REDECODE":
        Errors.append("R4 benchmark run did not reuse the complete cache")

    SampleRecords = []
    for Dataset in ("fma-small", "aime-open-model-subset"):
        DatasetRecords = [Record for Record in EligibleRecords if Record.Dataset == Dataset]
        for Index in (0, len(DatasetRecords) // 2, len(DatasetRecords) - 1):
            SampleRecords.append(DatasetRecords[Index])
    SampleEvidence = []
    for Record in SampleRecords:
        Key = CacheRecordKey(Record.Dataset, Record.SampleId, Record.Locator)
        IndexRow = IndexByKey[Key]
        NpyPath, _MetadataPath = CacheModule.CachePaths(Record, CacheRoot)
        Cached = np.load(NpyPath, allow_pickle=False)
        Raw = CacheModule.LoadRawRecordWaveform(Record, 16000).cpu().numpy()
        Raw = np.ascontiguousarray(Raw, dtype=np.float32)
        Exact = bool(np.array_equal(Cached, Raw))
        FileHashMatch = HashFile(NpyPath) == IndexRow["cache_file_sha256"]
        ValueHashMatch = (
            CacheModule.WaveformValueSha256(Cached)
            == IndexRow["waveform_value_sha256"]
        )
        if not Exact or not FileHashMatch or not ValueHashMatch:
            Errors.append(f"cache/raw exactness sample failed {Record.Dataset}/{Record.SampleId}")
        SampleEvidence.append(
            {
                "dataset": Record.Dataset,
                "sample_id": Record.SampleId,
                "sample_count": int(Cached.size),
                "raw_cache_array_equal": Exact,
                "cache_file_sha256_match": FileHashMatch,
                "waveform_value_sha256_match": ValueHashMatch,
                "maximum_absolute_delta": (
                    float(np.max(np.abs(Cached - Raw))) if Cached.size else 0.0
                ),
            }
        )
    QuarantineFiles = (
        sum(1 for FilePath in QuarantineRoot.rglob("*") if FilePath.is_file())
        if QuarantineRoot.is_dir()
        else 0
    )
    return {
        "cache_root": str(CacheRoot),
        "cache_index_sha256": IndexHash,
        "index_entries": len(IndexRows),
        "npy_file_count": len(NpyFiles),
        "sidecar_file_count": len(SidecarFiles),
        "npy_bytes": NpyBytes,
        "dtype_counts": dict(DtypeCounts),
        "dataset_counts": dict(DatasetCounts),
        "header_error_count": HeaderErrors,
        "sidecar_error_count": SidecarErrors,
        "manifest_file_sha_alignment_mismatch_count": ManifestHashMismatchCount,
        "raw_cache_exactness_samples": SampleEvidence,
        "raw_sources_retained": all(
            Path(re.sub(r"^(?:parquet://)?(.+?)(?:#row=\d+)?$", r"\1", Record.Locator)).exists()
            for Record in SampleRecords
        ),
        "quarantine_root": str(QuarantineRoot),
        "quarantine_file_count": QuarantineFiles,
        "active_index_quarantine_reference_count": QuarantineReferenceCount,
    }, Errors, IndexByKey


def AuditAime(
    AimeRows: Sequence[dict[str, str]],
    IndexByKey: Mapping[tuple[str, str, str], dict[str, Any]],
    RecordedResolution: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    import pyarrow.parquet as Parquet

    ByShard: dict[Path, list[tuple[dict[str, str], int]]] = defaultdict(list)
    for Row in AimeRows:
        Match = re.fullmatch(r"parquet://(.+)#row=(\d+)", Row["source_locator"])
        if Match is None:
            Errors.append(f"invalid AIME locator {Row['source_locator']}")
            continue
        ByShard[Path(Match.group(1))].append((Row, int(Match.group(2))))
    IdMismatchCount = 0
    IndexResolutionMismatchCount = 0
    for ShardPath, Items in ByShard.items():
        IdColumn = Parquet.read_table(ShardPath, columns=["id"]).column("id")
        for Row, DeclaredRow in Items:
            Resolved = DeclaredRow - 1
            if DeclaredRow < 1 or Resolved >= len(IdColumn):
                IdMismatchCount += 1
                continue
            ParquetId = str(IdColumn[Resolved].as_py())
            if ParquetId != Row["sample_id"]:
                IdMismatchCount += 1
            Key = CacheRecordKey(
                Row["dataset"], Row["sample_id"], Row["source_locator"]
            )
            IndexRow = IndexByKey.get(Key, {})
            if (
                IndexRow.get("declared_manifest_row") != DeclaredRow
                or IndexRow.get("resolved_zero_based_row") != Resolved
                or IndexRow.get("asserted_parquet_id") != Row["sample_id"]
                or IndexRow.get("source_audio_sha256") != Row["file_sha256"]
            ):
                IndexResolutionMismatchCount += 1
    if len(AimeRows) != 1009:
        Errors.append(f"non-test AIME record count {len(AimeRows)}")
    if len(ByShard) != 36:
        Errors.append(f"AIME shard count {len(ByShard)}")
    if IdMismatchCount:
        Errors.append(f"AIME one-based ID mismatches {IdMismatchCount}")
    if IndexResolutionMismatchCount:
        Errors.append(f"AIME index resolution/hash mismatches {IndexResolutionMismatchCount}")

    SampleRows = [
        AimeRows[(Index * len(AimeRows)) // 12]
        for Index in range(12)
    ]
    SampleByShard: dict[Path, list[tuple[dict[str, str], int]]] = defaultdict(list)
    for Row in SampleRows:
        Match = re.fullmatch(r"parquet://(.+)#row=(\d+)", Row["source_locator"])
        assert Match is not None
        SampleByShard[Path(Match.group(1))].append((Row, int(Match.group(2)) - 1))
    FileHashSamples = []
    for ShardPath, Items in SampleByShard.items():
        AudioColumn = Parquet.read_table(ShardPath, columns=["audio"]).column("audio")
        for Row, ResolvedRow in Items:
            AudioValue = AudioColumn[ResolvedRow].as_py()
            AudioBytes = AudioValue["bytes"]
            RawHash = Sha256Bytes(AudioBytes)
            Key = CacheRecordKey(
                Row["dataset"], Row["sample_id"], Row["source_locator"]
            )
            IndexHash = IndexByKey[Key]["source_audio_sha256"]
            Match = RawHash == Row["file_sha256"] == IndexHash
            if not Match:
                Errors.append(f"AIME raw file_sha256 sample mismatch {Row['sample_id']}")
            FileHashSamples.append(
                {
                    "sample_id": Row["sample_id"],
                    "raw_audio_sha256": RawHash,
                    "manifest_file_sha256": Row["file_sha256"],
                    "cache_index_source_audio_sha256": IndexHash,
                    "match": Match,
                }
            )
    ExpectedRecorded = {
        "status": "PASS",
        "resolver_version": "aime_manifest_one_based_v1",
        "scope": "non-test AIME records only",
        "checked_record_count": 1009,
        "checked_shard_count": 36,
        "declared_row_zero_count": 0,
        "minimum_declared_row": 1,
        "maximum_declared_row": 31,
        "parquet_id_mismatch_count": 0,
        "resolution_rule": "resolved_zero_based_row = declared_manifest_row - 1",
    }
    if RecordedResolution != ExpectedRecorded:
        Errors.append("recorded AIME resolver summary differs")
    return {
        "checked_record_count": len(AimeRows),
        "checked_shard_count": len(ByShard),
        "one_based_id_mismatch_count": IdMismatchCount,
        "cache_index_resolution_mismatch_count": IndexResolutionMismatchCount,
        "file_sha256_alignment_samples": FileHashSamples,
    }, Errors


def AuditWorkersAndSelection(
    Config: dict[str, Any],
    TrainingRecords: Sequence[Any],
    WorkerReport: dict[str, Any],
    GpuReport: dict[str, Any],
    Pilot: dict[str, Any],
    RunRecord: dict[str, Any],
    BenchmarkModule: Any,
    DeterminismModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    WorkerRows = WorkerReport.get("candidates", [])
    WorkerIds = [int(Row["workers"]) for Row in WorkerRows]
    if WorkerIds != [0, 2, 4]:
        Errors.append(f"worker candidates differ {WorkerIds}")
    if any(Row.get("status") != "PASS" for Row in WorkerRows):
        Errors.append("one or more worker candidates failed")
    TensorDigests = {Row["tensor_sequence_sha256"] for Row in WorkerRows}
    if len(TensorDigests) != 1 or not WorkerReport.get(
        "cross_worker_tensor_sequence_exact"
    ):
        Errors.append("worker 0/2/4 canonical tensor digests differ")
    _, WarmDigest = BenchmarkModule.ExpectedLocatorSequence(
        TrainingRecords, 1024, 20260830, 0
    )
    _, MeasuredDigest = BenchmarkModule.ExpectedLocatorSequence(
        TrainingRecords, 1024, 20260830, 1
    )
    if WorkerReport.get("expected_warm_locator_sequence_sha256") != WarmDigest:
        Errors.append("worker warm expected locator digest differs")
    if WorkerReport.get("expected_measured_locator_sequence_sha256") != MeasuredDigest:
        Errors.append("worker measured expected locator digest differs")
    if any(Row["warm_locator_sequence_sha256"] != WarmDigest for Row in WorkerRows):
        Errors.append("worker warm locator sequence differs")
    if any(Row["locator_sequence_sha256"] != MeasuredDigest for Row in WorkerRows):
        Errors.append("worker measured locator sequence differs")
    if any(
        Row.get("parent_intraop_threads") != 1
        or Row.get("parent_interop_threads") != 1
        or Row.get("worker_intraop_threads") != 1
        for Row in WorkerRows
    ):
        Errors.append("worker CPU thread pin evidence differs from 1/1/1")
    WorkerSource = inspect.getsource(DeterminismModule.PinDataLoaderWorkerCpuThreads)
    if "torch.set_num_threads(1)" not in WorkerSource:
        Errors.append("worker init does not pin torch CPU threads to one")
    ExpectedWorker = max(
        WorkerRows,
        key=lambda Row: (float(Row["measured_samples_per_second"]), -int(Row["workers"])),
    )
    if WorkerReport.get("recommended_workers") != ExpectedWorker["workers"]:
        Errors.append("worker recommendation is not measured-throughput maximum")

    GpuRows = [
        Row
        for Row in GpuReport.get("candidates", [])
        if Row.get("status") == "PASS" and float(Row["memory_fraction"]) <= 0.85
    ]
    ExpectedBatch = max(
        GpuRows,
        key=lambda Row: (float(Row["segments_per_second"]), int(Row["batch_size"])),
    )
    if GpuReport.get("recommended_batch_size") != ExpectedBatch["batch_size"]:
        Errors.append("GPU batch recommendation is not measured-throughput maximum")
    if GpuReport.get("precision_mode") != "fp32_guarded" or GpuReport.get(
        "autocast_enabled"
    ) is not False or GpuReport.get("grad_scaler_enabled") is not False:
        Errors.append("GPU autotune is not guarded FP32")
    SelectedConfig = RunRecord["config"]
    if SelectedConfig.get("workers") != ExpectedWorker["workers"]:
        Errors.append("run config worker selection differs")
    if SelectedConfig.get("batch_size") != ExpectedBatch["batch_size"]:
        Errors.append("run config batch selection differs")
    if Pilot.get("workers") != ExpectedWorker["workers"] or Pilot.get(
        "batch_size"
    ) != ExpectedBatch["batch_size"]:
        Errors.append("cached pilot did not use selected worker/batch")
    return {
        "worker_candidates": WorkerRows,
        "cross_worker_tensor_digest": next(iter(TensorDigests)) if TensorDigests else None,
        "independent_warm_locator_sha256": WarmDigest,
        "independent_measured_locator_sha256": MeasuredDigest,
        "selected_workers": ExpectedWorker["workers"],
        "selected_worker_measured_samples_per_second": ExpectedWorker[
            "measured_samples_per_second"
        ],
        "selected_batch_size": ExpectedBatch["batch_size"],
        "selected_batch_segments_per_second": ExpectedBatch["segments_per_second"],
        "cpu_thread_pin_source_verified": "torch.set_num_threads(1)" in WorkerSource,
    }, Errors


def AuditProjection(
    Config: dict[str, Any],
    ValidationRows: Sequence[dict[str, str]],
    Pilot: dict[str, Any],
    Projection: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    SamplesPerEpoch = int(Config["samples_per_epoch"])
    GroupProduct = int(Config["balanced_group_draws_per_epoch"]) * int(
        Config["samples_per_balanced_group_draw"]
    )
    TrainingPerSeed = SamplesPerEpoch * int(Config["epochs"])
    TrainingThreeSeeds = TrainingPerSeed * len(Config["seeds"])
    ValidationSegments = sum(
        min(
            int(Config["max_segments_per_file"]),
            max(
                1,
                math.ceil(
                    float(Row["duration_seconds"])
                    / float(Config["segment_seconds"])
                ),
            ),
        )
        for Row in ValidationRows
    )
    MeasuredRate = float(Pilot["end_to_end_samples_per_second"])
    SafetyFactor = float(Config["runtime_projection_safety_factor"])
    ConservativeRate = MeasuredRate * SafetyFactor
    PerSeedHours = (TrainingPerSeed + ValidationSegments) / ConservativeRate / 3600.0
    ThreeSeedHours = PerSeedHours * len(Config["seeds"])
    Expected = {
        "samples_per_epoch": SamplesPerEpoch,
        "epochs": int(Config["epochs"]),
        "seed_count": len(Config["seeds"]),
        "training_samples_per_seed": TrainingPerSeed,
        "training_samples_three_seeds": TrainingThreeSeeds,
        "validation_segments_per_seed": ValidationSegments,
        "measured_cached_end_to_end_samples_per_second": MeasuredRate,
        "projection_safety_factor": SafetyFactor,
        "conservative_samples_per_second": ConservativeRate,
        "projected_hours_per_seed_including_validation": PerSeedHours,
        "projected_three_seed_wall_hours": ThreeSeedHours,
        "full_training_wall_gate_hours": float(Config["full_training_wall_gate_hours"]),
    }
    if SamplesPerEpoch != GroupProduct:
        Errors.append("samples_per_epoch differs from group-draw product")
    for Field, ExpectedValue in Expected.items():
        Found = float(Projection[Field])
        if not math.isclose(Found, float(ExpectedValue), rel_tol=0.0, abs_tol=1e-12):
            Errors.append(f"projection {Field} differs: {Found} != {ExpectedValue}")
    Ready = ThreeSeedHours <= float(Config["full_training_wall_gate_hours"])
    if Projection.get("status") != (
        "READY_FOR_FULL_TRAINING" if Ready else "BLOCKED_RESOURCE"
    ):
        Errors.append("projection readiness status differs")
    if Projection.get("statistical_workload_reduced") is not False:
        Errors.append("projection says statistical workload was reduced")
    return {
        **Expected,
        "group_draw_product": GroupProduct,
        "ready_under_24_hour_gate": Ready,
        "statistical_workload_reduced": Projection.get(
            "statistical_workload_reduced"
        ),
    }, Errors


def AuditOutputScope(
    DeepvoiceRoot: Path,
    RunRecord: dict[str, Any],
    ReportText: str,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    ArtifactRoot = DeepvoiceRoot / "artifacts" / "e01_r4"
    ArtifactFiles = sorted(
        str(FilePath.relative_to(DeepvoiceRoot)).replace("\\", "/")
        for FilePath in ArtifactRoot.rglob("*")
        if FilePath.is_file()
    ) if ArtifactRoot.is_dir() else []
    CheckpointRoot = DeepvoiceRoot / "checkpoints" / "e01_r4"
    CheckpointFiles = sorted(
        str(FilePath.relative_to(DeepvoiceRoot)).replace("\\", "/")
        for FilePath in CheckpointRoot.rglob("*")
        if FilePath.is_file()
    ) if CheckpointRoot.is_dir() else []
    ForbiddenReports = [
        DeepvoiceRoot / "reports" / "e01-r4-results.json",
        DeepvoiceRoot / "reports" / "e01-r4-metrics.json",
        DeepvoiceRoot / "reports" / "e01-r4-validation-oof.csv.gz",
    ]
    if ArtifactFiles != ["artifacts/e01_r4/benchmark-run.json"]:
        Errors.append(f"unexpected R4 artifact files {ArtifactFiles}")
    if CheckpointFiles:
        Errors.append(f"R4 checkpoint files exist {CheckpointFiles}")
    if any(FilePath.exists() for FilePath in ForbiddenReports):
        Errors.append("R4 full-training result/metric/OOF report exists")
    if RunRecord.get("full_training_started") is not False:
        Errors.append("R4 full_training_started is not false")
    if RunRecord.get("experiment_batch") != "READY_FOR_FULL_TRAINING":
        Errors.append("R4 experiment batch status differs")
    if RunRecord.get("execution_phase") != (
        "CACHE_AND_BENCHMARK_ONLY_BEFORE_LONG_FULL_TRAINING"
    ):
        Errors.append("R4 execution phase is not benchmark-only")
    RequiredReportClaims = (
        "full 3-seed training started: false",
        "validation OOF/metric/checkpoint: 생성하지 않음",
        "E02: 실행하지 않음",
        "R1/R2/R3 code와 reports: 보존",
    )
    for Claim in RequiredReportClaims:
        if Claim not in ReportText:
            Errors.append(f"missing scope claim: {Claim}")
    return {
        "artifact_files": ArtifactFiles,
        "checkpoint_files": CheckpointFiles,
        "forbidden_report_count": sum(
            FilePath.exists() for FilePath in ForbiddenReports
        ),
        "full_training_started": RunRecord.get("full_training_started"),
        "execution_phase": RunRecord.get("execution_phase"),
        "e02_absence_claim": "E02: 실행하지 않음" in ReportText,
    }, Errors


def WriteOutputs(OutputRoot: Path, Record: dict[str, Any]) -> None:
    OutputRoot.mkdir(parents=True, exist_ok=False)
    JsonPath = OutputRoot / "e01-r4-validation-audit.json"
    MarkdownPath = OutputRoot / "e01-r4-validation-audit.md"
    JsonPath.write_text(
        json.dumps(Record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Lines = [
        f"EXPERIMENT_AUDIT: {Record['status']}",
        "",
        "# DeepVoice E01-R4 independent validation audit",
        "",
        "## 판정",
        "",
        "E01-R4의 cache·수치 guard·worker·projection 증거를 독립 재검산했고 `READY_FOR_FULL_TRAINING` benchmark 판정을 승인한다. 이 판정은 장시간 학습 결과나 validation 성능 승인이 아니다.",
        "",
    ]
    Failed = [Check for Check in Record["checks"] if Check["status"] != "PASS"]
    if Failed:
        Lines.extend(["## 차단 사유", ""])
        for Check in Failed:
            for Error in Check["errors"]:
                Lines.append(f"- {Check['check']}: {Error}")
        Lines.append("")
    Lines.extend(
        [
            "## 독립 통과 사항",
            "",
            "- R4 report/artifact JSON 13개와 cache control/index/sidecar를 strict recursive-finite 방식으로 검사했고 nested NaN hard reject를 재현했다.",
            "- 15/15 unit/adversarial tests를 새로 실행했으며 masked NaN, skipped optimizer step, FP32-only, worker 0/2/4 검사가 포함된다.",
            "- fresh CUDA guarded smoke와 32-batch cached pilot에서 FP32 logits/loss/gradient/parameter/step guard, skip 0을 확인했다.",
            "- R4 source inventory 16개 byte/SHA와 digest가 run manifest에 일치한다.",
            "- manifest SHA, 137,328 rows, split crossing 0, test crossing-only 격리를 확인했다.",
            "- active cache 5,651 index/NPY/sidecar, 9,557,290,108 NPY bytes, float32 1-D sample counts를 전수 검사했다.",
            "- cache/raw exact array sample 6개, AIME 1,009 ID 전수 및 raw audio file_sha256 sample 12개를 독립 검증했다.",
            "- invalid quarantine는 active root 밖에 있으며 active index reference는 0이다. 표본 raw source hash는 manifest와 일치한다.",
            "- worker 0/2/4 canonical tensor digest와 locator sequence가 동일하고 CPU thread pin 1/1/1, workers=2 선택을 확인했다.",
            "- batch=32 선택과 cached end-to-end 98.6615 sample/s, safety factor 0.80을 확인했다.",
            "- 32,768×20×3=1,966,080 training samples와 validation 18,165 segments/seed를 포함한 7.111063987시간을 재계산했다.",
            "- full training/checkpoint/validation OOF/metric/E02 산출물은 없다.",
            "",
            "## 승인 범위",
            "",
            "다음 단계는 고정된 R4 code inventory와 cache index hash를 다시 gate한 뒤 full 3-seed training을 별도 immutable run으로 시작하는 것이다. 이 감사 자체는 학습 시작, checkpoint 생성 또는 성능 주장을 수행하지 않았다.",
        ]
    )
    MarkdownPath.write_text("\n".join(Lines) + "\n", encoding="utf-8")


def Execute(RepoRoot: Path, OutputRoot: Path) -> dict[str, Any]:
    Started = time.perf_counter()
    DeepvoiceRoot = RepoRoot / "deepvoice"
    SourceRoot = DeepvoiceRoot / "experiments" / "e01_r4"
    ReportsRoot = DeepvoiceRoot / "reports"
    CacheRoot = DeepvoiceRoot / "data" / "cache" / "e01_r4"
    QuarantineRoot = (
        DeepvoiceRoot / "data" / "cache" / "e01_r4_invalid_quarantine_20260830"
    )
    ManifestPath = ReportsRoot / "deepvoice-training-manifest.csv.gz"
    Config = LoadStrictJson(SourceRoot / "config.json")
    RunRecord = LoadStrictJson(ReportsRoot / "e01-r4-run-manifest.json")
    RecordedTests = LoadStrictJson(ReportsRoot / "e01-r4-unit-test-results.json")
    Pilot = LoadStrictJson(ReportsRoot / "e01-r4-cached-gpu-pilot.json")
    WorkerReport = LoadStrictJson(ReportsRoot / "e01-r4-windows-workers.json")
    GpuReport = LoadStrictJson(ReportsRoot / "e01-r4-batch-autotune.json")
    Projection = LoadStrictJson(ReportsRoot / "e01-r4-runtime-projection.json")
    CacheIntegrity = LoadStrictJson(ReportsRoot / "e01-r4-cache-integrity.json")
    CacheBuild = LoadStrictJson(ReportsRoot / "e01-r4-cache-build.json")
    AimeResolution = LoadStrictJson(ReportsRoot / "e01-r4-aime-row-resolution.json")
    ReportText = (ReportsRoot / "e01-r4-experiment-batch.md").read_text(
        encoding="utf-8"
    )

    if str(DeepvoiceRoot) not in sys.path:
        sys.path.insert(0, str(DeepvoiceRoot))
    from experiments.e01_r4 import benchmark as BenchmarkModule
    from experiments.e01_r4 import cache as CacheModule
    from experiments.e01_r4 import determinism as DeterminismModule
    from experiments.e01_r4 import numerical as NumericalModule
    from experiments.e01_r4 import records as RecordsModule
    from experiments.e01_r4 import run_tests as RunTestsModule
    from experiments.e01_r4 import strict_serialization as SerializationModule
    from experiments.e01_r4 import train_e01 as TrainModule

    Checks: list[dict[str, Any]] = []
    ManifestEvidence, ManifestErrors, TrainingRows, ValidationRows = AuditManifest(
        ManifestPath, RunRecord, RecordsModule
    )
    AddCheck(Checks, "manifest_and_test_isolation", ManifestErrors, ManifestEvidence)

    InventoryEvidence, InventoryErrors = AuditInventory(
        SourceRoot,
        ReportsRoot / "e01-r4-code-inventory.csv",
        RunRecord,
    )
    AddCheck(Checks, "immutable_source_inventory", InventoryErrors, InventoryEvidence)

    StrictEvidence, StrictErrors = AuditStrictJson(
        ReportsRoot,
        DeepvoiceRoot / "artifacts" / "e01_r4" / "benchmark-run.json",
        CacheRoot,
        SerializationModule,
    )
    AddCheck(Checks, "strict_recursive_json", StrictErrors, StrictEvidence)

    TestEvidence, TestErrors = AuditTestsAndNumerics(
        Config,
        RecordedTests,
        Pilot,
        RunTestsModule,
        NumericalModule,
        BenchmarkModule,
        TrainModule,
    )
    AddCheck(Checks, "tests_and_guarded_fp32", TestErrors, TestEvidence)

    TrainingRecords, ValidationRecords, _Summary = RecordsModule.LoadE01Records(
        ManifestPath
    )
    EligibleRecords = [
        Record
        for Record in TrainingRecords + ValidationRecords
        if Record.Dataset in ("fma-small", "aime-open-model-subset")
    ]
    CacheEvidence, CacheErrors, IndexByKey = AuditCache(
        CacheRoot,
        QuarantineRoot,
        EligibleRecords,
        TrainingRows + ValidationRows,
        CacheIntegrity,
        CacheBuild,
        CacheModule,
    )
    AddCheck(Checks, "exact_cache_and_quarantine", CacheErrors, CacheEvidence)

    AimeRows = [
        Row
        for Row in TrainingRows + ValidationRows
        if Row["dataset"] == "aime-open-model-subset"
    ]
    AimeEvidence, AimeErrors = AuditAime(
        AimeRows, IndexByKey, AimeResolution
    )
    AddCheck(Checks, "aime_one_based_resolution", AimeErrors, AimeEvidence)

    WorkerEvidence, WorkerErrors = AuditWorkersAndSelection(
        Config,
        TrainingRecords,
        WorkerReport,
        GpuReport,
        Pilot,
        RunRecord,
        BenchmarkModule,
        DeterminismModule,
    )
    AddCheck(Checks, "workers_batch_and_measured_selection", WorkerErrors, WorkerEvidence)

    ProjectionEvidence, ProjectionErrors = AuditProjection(
        Config, ValidationRows, Pilot, Projection
    )
    AddCheck(Checks, "runtime_projection_math", ProjectionErrors, ProjectionEvidence)

    OutputEvidence, OutputErrors = AuditOutputScope(
        DeepvoiceRoot, RunRecord, ReportText
    )
    AddCheck(Checks, "benchmark_only_output_scope", OutputErrors, OutputEvidence)

    Status = "PASS" if all(Check["status"] == "PASS" for Check in Checks) else "BLOCKED"
    Record = {
        "status": Status,
        "audit_id": "E01-R4-INDEPENDENT-AUDIT-20260830",
        "scope": "experiments/e01_r4 and e01-r4-* benchmark-only readiness",
        "test_data_handling": (
            "test rows used only as group/split crossing projections; no test statistics, "
            "predictions, labels, masks, or metrics calculated"
        ),
        "checks": Checks,
        "input_hashes": {
            "manifest": HashFile(ManifestPath),
            "run_manifest": HashFile(ReportsRoot / "e01-r4-run-manifest.json"),
            "code_inventory": HashFile(ReportsRoot / "e01-r4-code-inventory.csv"),
            "cache_index": HashFile(CacheRoot / "cache-index.jsonl"),
            "batch_report": HashFile(ReportsRoot / "e01-r4-experiment-batch.md"),
        },
        "audit_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "parent_intraop_threads": torch.get_num_threads(),
            "parent_interop_threads": torch.get_num_interop_threads(),
        },
        "runtime_seconds": time.perf_counter() - Started,
    }
    WriteOutputs(OutputRoot, Record)
    return Record


def Main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_e01_r4.py REPO_ROOT OUTPUT_ROOT")
    Record = Execute(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    print(
        json.dumps(
            {
                "status": Record["status"],
                "checks": [
                    {
                        "check": Check["check"],
                        "status": Check["status"],
                        "errors": Check["errors"],
                    }
                    for Check in Record["checks"]
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if Record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(Main())
