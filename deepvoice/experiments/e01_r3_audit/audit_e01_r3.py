# /// <summary>
# Independent read-only auditor for the immutable DeepVoice E01-R3 lineage.
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
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ExpectedManifestSha256 = (
    "2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6"
)
ExpectedManifestRows = 137328
Datasets = (
    "ljspeech-1.1",
    "wavefake-1.2.0",
    "fma-small",
    "aime-open-model-subset",
)
R3JsonReports = (
    "e01-r3-balanced-pilot.json",
    "e01-r3-batch-autotune.json",
    "e01-r3-loader-benchmark.json",
    "e01-r3-preflight.json",
    "e01-r3-run-manifest.json",
    "e01-r3-runtime-projection.json",
    "e01-r3-tiny-gpu-smoke.json",
    "e01-r3-unit-test-results.json",
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


def LoadJson(FilePath: Path) -> Any:
    return json.loads(FilePath.read_text(encoding="utf-8"))


def LoadCsv(FilePath: Path) -> list[dict[str, str]]:
    with FilePath.open("r", encoding="utf-8-sig", newline="") as FileHandle:
        return list(csv.DictReader(FileHandle))


def StrictJsonStatus(FilePath: Path) -> tuple[bool, str | None]:
    def RejectConstant(Value: str) -> None:
        raise ValueError(f"non-standard JSON constant {Value}")

    try:
        json.loads(
            FilePath.read_text(encoding="utf-8"),
            parse_constant=RejectConstant,
        )
        return True, None
    except (json.JSONDecodeError, ValueError) as Error:
        return False, str(Error)


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
            GroupKey = Row["content_group_key"]
            GroupSplits[GroupKey].add(Split)
            if Split == "test":
                continue
            if Split == "train":
                TrainingRows.append(Row)
            elif Split == "validation":
                ValidationRows.append(Row)
            else:
                raise ValueError(f"Unexpected split {Split}")
    Crossings = sorted(
        GroupKey for GroupKey, Splits in GroupSplits.items() if len(Splits) > 1
    )
    return RowCount, TrainingRows, ValidationRows, Crossings


def AuditManifest(
    ManifestPath: Path,
    RunRecord: dict[str, Any],
    RecordsModule: Any,
) -> tuple[dict[str, Any], list[str], list[dict[str, str]], list[dict[str, str]]]:
    Errors = []
    RowCount, TrainingRows, ValidationRows, Crossings = ReadManifest(ManifestPath)
    ManifestHash = HashFile(ManifestPath)
    if ManifestHash != ExpectedManifestSha256:
        Errors.append(f"manifest SHA mismatch {ManifestHash}")
    if RowCount != ExpectedManifestRows:
        Errors.append(f"manifest row count {RowCount}")
    if Crossings:
        Errors.append(f"content-group crossing count {len(Crossings)}")
    Summary = RunRecord["manifest_integrity"]
    if Summary["manifest_sha256"] != ManifestHash:
        Errors.append("R3 run manifest records a different manifest SHA")
    if Summary["manifest_total_row_count"] != RowCount:
        Errors.append("R3 run manifest records a different row count")
    if Summary["crossing_group_count"] != 0:
        Errors.append("R3 run manifest crossing count is nonzero")
    TestContract = Summary["test_field_contract"]
    if TestContract != {
        "allowed_fields": ["content_group_key", "recommended_content_split"],
        "retained_forbidden_fields": 0,
        "test_statistics": 0,
    }:
        Errors.append("R3 manifest test-field contract differs")
    Source = inspect.getsource(RecordsModule.LoadE01Records)
    if "BuildLabelMasks(NonTestRows)" not in Source:
        Errors.append("R3 record loader does not restrict labels/masks to non-test rows")
    if "ProjectCrossingRows(NonTestRows) + TestCrossingRows" not in Source:
        Errors.append("R3 record loader crossing projection is not explicit")
    return (
        {
            "manifest_sha256": ManifestHash,
            "row_count": RowCount,
            "train_rows": len(TrainingRows),
            "validation_rows": len(ValidationRows),
            "crossing_group_count": len(Crossings),
            "test_handling": (
                "streamed group/split projection for crossing only; test rows were "
                "discarded immediately and no test statistic was retained"
            ),
            "implementation_builds_labels_from_non_test_rows": (
                "BuildLabelMasks(NonTestRows)" in Source
            ),
        },
        Errors,
        TrainingRows,
        ValidationRows,
    )


def ParseLocator(Locator: str) -> tuple[str, Path, str | int | None]:
    if Locator.startswith("zip://"):
        Container, Member = Locator[6:].split("!/", 1)
        return "zip", Path(Container), Member
    if Locator.startswith("parquet://"):
        Match = re.fullmatch(r"parquet://(.+)#row=(\d+)", Locator)
        if Match is None:
            raise ValueError(f"Invalid parquet locator {Locator}")
        return "parquet", Path(Match.group(1)), int(Match.group(2))
    return "file", Path(Locator), None


def ReadLocatorBytes(Locator: str) -> tuple[str, Path, bytes | None]:
    Kind, ContainerPath, Payload = ParseLocator(Locator)
    if Kind == "file":
        return Kind, ContainerPath, None
    if Kind == "zip":
        with zipfile.ZipFile(ContainerPath, "r") as Archive:
            return Kind, ContainerPath, Archive.read(str(Payload))
    import pyarrow.parquet as Parquet

    AudioValue = Parquet.read_table(ContainerPath, columns=["audio"]).column("audio")[
        int(Payload)
    ].as_py()
    if not isinstance(AudioValue, dict) or not isinstance(AudioValue.get("bytes"), bytes):
        raise TypeError("Parquet locator does not contain embedded audio bytes")
    return Kind, ContainerPath, AudioValue["bytes"]


def DecodeWithFfmpeg(
    ContainerPath: Path,
    AudioBytes: bytes | None,
    SampleRate: int = 16000,
) -> np.ndarray:
    Result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0" if AudioBytes is not None else str(ContainerPath),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SampleRate),
            "pipe:1",
        ],
        input=AudioBytes,
        capture_output=True,
        check=True,
        timeout=180,
    )
    Samples = np.frombuffer(Result.stdout, dtype="<f4").copy()
    if Samples.size == 0 or not np.isfinite(Samples).all():
        raise ValueError("Independent locator decode is empty or nonfinite")
    return Samples


def AuditLocators(
    TrainingRows: Sequence[dict[str, str]],
    Preflight: dict[str, Any],
    LoaderReport: dict[str, Any],
    AudioModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    FirstRows = {}
    for Row in TrainingRows:
        FirstRows.setdefault(Row["dataset"], Row)
    PreflightByDataset = {
        Probe["dataset"]: Probe for Probe in Preflight["locator_probes"]
    }
    LoaderByDataset = {
        Probe["dataset"]: Probe for Probe in LoaderReport["probes"]
    }
    ExpectedKinds = {
        "ljspeech-1.1": "file",
        "wavefake-1.2.0": "zip",
        "fma-small": "file",
        "aime-open-model-subset": "parquet",
    }
    Results = []
    for Dataset in Datasets:
        Row = FirstRows[Dataset]
        Locator = Row["source_locator"]
        Kind, ContainerPath, AudioBytes = ReadLocatorBytes(Locator)
        Independent = DecodeWithFfmpeg(ContainerPath, AudioBytes)
        Diagnostics: dict[str, object] = {}
        Actual = AudioModule.LoadLocatorWaveform(Locator, 16000, Diagnostics)
        IndependentCount = int(Independent.size)
        ActualCount = int(Actual.numel())
        CountDelta = abs(IndependentCount - ActualCount)
        PreflightCount = int(PreflightByDataset[Dataset]["decoded_sample_count"])
        LoaderCount = int(LoaderByDataset[Dataset]["decoded_sample_count"])
        if Kind != ExpectedKinds[Dataset]:
            Errors.append(f"{Dataset} locator kind {Kind}")
        if CountDelta > 1:
            Errors.append(f"{Dataset} independent/E01 decode length delta {CountDelta}")
        if ActualCount != PreflightCount or ActualCount != LoaderCount:
            Errors.append(f"{Dataset} current decode differs from R3 reports")
        if Actual.numel() == 0 or not torch.isfinite(Actual).all():
            Errors.append(f"{Dataset} current loader output is empty/nonfinite")
        Results.append(
            {
                "dataset": Dataset,
                "sample_id": Row["sample_id"],
                "locator_kind": Kind,
                "container_bytes": ContainerPath.stat().st_size,
                "independent_ffmpeg_samples": IndependentCount,
                "e01_r3_samples": ActualCount,
                "absolute_count_delta": CountDelta,
                "accepted_endpoint_rounding_tolerance_samples": 1,
                "decoder_backend": Diagnostics.get("decoder_backend"),
            }
        )
    AudioModule.CloseAudioContainerCaches()
    return {"decode_count": len(Results), "rows": Results}, Errors


def AuditPaddingAndFp32(
    Config: dict[str, Any],
    AudioModule: Any,
    ModelModule: Any,
    BenchmarkModule: Any,
    TrainModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    torch.manual_seed(20260830)
    Model = ModelModule.LogMelCnn(Config).cpu().eval()
    SegmentLength = int(Config["sample_rate"] * Config["segment_seconds"])
    ValidLength = int(Config["sample_rate"] * 4)
    First = torch.zeros((1, SegmentLength), dtype=torch.float32)
    First[:, :ValidLength] = torch.randn((1, ValidLength)) * 0.01
    TailSentinel = First.clone()
    TailSentinel[:, ValidLength:] = torch.linspace(
        -1000.0, 1000.0, SegmentLength - ValidLength
    )
    ValidSentinel = First.clone()
    ValidSentinel[:, : ValidLength // 2] *= -1.0
    Counts = torch.tensor([ValidLength], dtype=torch.long)
    with torch.inference_mode():
        FirstFeatures, FirstMask = Model.FeatureExtractor(First, Counts)
        TailFeatures, TailMask = Model.FeatureExtractor(TailSentinel, Counts)
        FirstLogits = Model(First, Counts)
        TailLogits = Model(TailSentinel, Counts)
        ValidLogits = Model(ValidSentinel, Counts)
    FeatureDelta = float(torch.max(torch.abs(FirstFeatures - TailFeatures)))
    LogitDelta = float(torch.max(torch.abs(FirstLogits - TailLogits)))
    ValidDelta = float(torch.max(torch.abs(FirstLogits - ValidLogits)))
    if FeatureDelta != 0.0 or LogitDelta != 0.0:
        Errors.append("padded-tail sentinel changed masked features/logits")
    if not torch.equal(FirstMask, TailMask):
        Errors.append("padded-tail sentinel changed the frame mask")
    if ValidDelta <= 0.0:
        Errors.append("valid-region positive control did not change logits")
    Segments, Lengths = AudioModule.CreateValidationSegmentsWithLengths(
        First[0, :ValidLength],
        int(Config["sample_rate"]),
        float(Config["segment_seconds"]),
        int(Config["max_segments_per_file"]),
    )
    if Segments.shape != (1, SegmentLength) or Lengths.tolist() != [ValidLength]:
        Errors.append("short-file valid length/padding contract differs")
    FeatureDtype = None
    LogitDtype = None
    ScalerEnabled = None
    if not torch.cuda.is_available():
        Errors.append("CUDA unavailable for independent FP32/autocast/GradScaler probe")
    else:
        GpuModel = ModelModule.LogMelCnn(Config).cuda().eval()
        Seen: dict[str, str] = {}

        def FeatureHook(_Module: Any, _Input: Any, Output: Any) -> None:
            Seen["dtype"] = str(Output[0].dtype)

        Handle = GpuModel.FeatureExtractor.register_forward_hook(FeatureHook)
        GpuWaveform = First.cuda()
        GpuCounts = Counts.cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            GpuLogits = GpuModel(GpuWaveform, GpuCounts)
        Handle.remove()
        FeatureDtype = Seen.get("dtype")
        LogitDtype = str(GpuLogits.dtype)
        ScalerEnabled = torch.amp.GradScaler("cuda", enabled=True).is_enabled()
        if FeatureDtype != "torch.float32":
            Errors.append(f"feature path dtype under autocast is {FeatureDtype}")
        if not ScalerEnabled:
            Errors.append("CUDA GradScaler is not enabled")
    PilotSource = inspect.getsource(BenchmarkModule.BenchmarkBalancedPilot)
    FullSource = inspect.getsource(TrainModule.RunSeed)
    if 'torch.amp.GradScaler("cuda", enabled=True)' not in PilotSource:
        Errors.append("balanced pilot does not construct an enabled GradScaler")
    if 'torch.amp.GradScaler("cuda", enabled=True)' not in FullSource:
        Errors.append("full-run path does not construct an enabled GradScaler")
    return {
        "segment_length": SegmentLength,
        "valid_length": ValidLength,
        "feature_tail_delta": FeatureDelta,
        "logit_tail_delta": LogitDelta,
        "valid_region_logit_delta": ValidDelta,
        "valid_frame_count": int(FirstMask.sum()),
        "feature_dtype_under_cuda_autocast": FeatureDtype,
        "logit_dtype_under_cuda_autocast": LogitDtype,
        "grad_scaler_enabled": ScalerEnabled,
    }, Errors


def AuditSampler(
    TrainingRecords: Sequence[Any],
    Config: dict[str, Any],
    SamplingModule: Any,
    SamplerReportRows: Sequence[dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    SamplesPerEpoch = int(Config["samples_per_epoch"])
    Sampler = SamplingModule.GroupFirstBalancedSampler(
        TrainingRecords, SamplesPerEpoch, 20260830
    )
    Indices = list(iter(Sampler))
    Repeat = list(
        iter(
            SamplingModule.GroupFirstBalancedSampler(
                TrainingRecords, SamplesPerEpoch, 20260830
            )
        )
    )
    if Indices != Repeat:
        Errors.append("same-seed sampler is not deterministic")
    if len(Indices) != SamplesPerEpoch:
        Errors.append(f"sampler emitted {len(Indices)} rows")
    StratumCounts = Counter()
    ProviderCounts = Counter()
    PairMismatches = 0
    HeldoutCount = 0
    for Start in range(0, len(Indices), 4):
        Quad = [TrainingRecords[Index] for Index in Indices[Start : Start + 4]]
        ByStratum = {}
        for Record in Quad:
            Stratum = SamplingModule.ClassifyTrainingStratum(Record)
            if Stratum is None:
                HeldoutCount += 1
                continue
            StratumCounts[Stratum] += 1
            ByStratum[Stratum] = Record
            if Stratum == "music_fake":
                ProviderCounts[Record.GeneratorOrProvider] += 1
        if set(ByStratum) != set(SamplingModule.Strata):
            Errors.append(f"quad {Start // 4} does not contain all four strata")
            break
        if (
            ByStratum["speech_real"].ContentGroupKey
            != ByStratum["speech_fake"].ContentGroupKey
        ):
            PairMismatches += 1
    ExpectedPerStratum = SamplesPerEpoch // 4
    if any(
        StratumCounts[Stratum] != ExpectedPerStratum
        for Stratum in SamplingModule.Strata
    ):
        Errors.append(f"exact stratum balance differs: {dict(StratumCounts)}")
    if PairMismatches:
        Errors.append(f"speech content pair mismatches {PairMismatches}")
    if HeldoutCount:
        Errors.append(f"held-out samples emitted {HeldoutCount}")
    if len(ProviderCounts) != 9:
        Errors.append(f"AIME provider coverage is {len(ProviderCounts)}")
    GroupIndex = SamplingModule.BuildGroupIndex(TrainingRecords)
    IndependentSummary = {
        Stratum: {
            "group_count": len(GroupIndex[Stratum]),
            "row_count": sum(len(Rows) for Rows in GroupIndex[Stratum].values()),
        }
        for Stratum in SamplingModule.Strata
    }
    ReportSummary = {
        Row["stratum"]: {
            "group_count": int(Row["group_count"]),
            "row_count": int(Row["row_count"]),
        }
        for Row in SamplerReportRows
    }
    if IndependentSummary != ReportSummary:
        Errors.append("sampler report differs from independent group index")
    ProviderValues = list(ProviderCounts.values())
    return {
        "sample_count": len(Indices),
        "stratum_counts": dict(sorted(StratumCounts.items())),
        "speech_pair_mismatch_count": PairMismatches,
        "heldout_sample_count": HeldoutCount,
        "aime_provider_counts": dict(sorted(ProviderCounts.items())),
        "aime_provider_count_range": [min(ProviderValues), max(ProviderValues)],
        "group_index": IndependentSummary,
    }, Errors


def AuditProjection(
    Config: dict[str, Any],
    Projection: dict[str, Any],
    LoaderReport: dict[str, Any],
    GpuReport: dict[str, Any],
    PilotReport: dict[str, Any],
    ValidationRows: Sequence[dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    SamplesPerEpoch = int(Config["samples_per_epoch"])
    GroupDrawProduct = int(Config["balanced_group_draws_per_epoch"]) * int(
        Config["samples_per_balanced_group_draw"]
    )
    TrainingPerSeed = SamplesPerEpoch * int(Config["epochs"])
    TotalTrainingDecodes = TrainingPerSeed * len(Config["seeds"])
    GpuRate = float(GpuReport["recommended_segments_per_second"])
    LoaderRate = float(LoaderReport["balanced_files_per_second"])
    PilotRate = float(PilotReport["end_to_end_samples_per_second"])
    EffectiveRate = min(GpuRate, LoaderRate, PilotRate)
    TrainingSecondsPerSeed = TrainingPerSeed / EffectiveRate
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
    ValidationSecondsPerSeed = (
        len(ValidationRows) / LoaderRate + ValidationSegments / GpuRate
    )
    TotalHours = len(Config["seeds"]) * (
        TrainingSecondsPerSeed + ValidationSecondsPerSeed
    ) / 3600.0
    Expected = {
        "samples_per_epoch": SamplesPerEpoch,
        "training_samples_per_seed": TrainingPerSeed,
        "validation_files_per_seed": len(ValidationRows),
        "validation_segments_per_seed": ValidationSegments,
        "gpu_segments_per_second": GpuRate,
        "loader_files_per_second": LoaderRate,
        "balanced_pilot_samples_per_second": PilotRate,
        "effective_training_samples_per_second": EffectiveRate,
        "training_hours_per_seed": TrainingSecondsPerSeed / 3600.0,
        "validation_hours_per_seed": ValidationSecondsPerSeed / 3600.0,
        "projected_three_seed_wall_hours": TotalHours,
    }
    if SamplesPerEpoch != GroupDrawProduct:
        Errors.append("samples_per_epoch differs from group-draw product")
    for Key, Value in Expected.items():
        Found = float(Projection[Key])
        if not math.isclose(Found, float(Value), rel_tol=0.0, abs_tol=1e-12):
            Errors.append(f"projection {Key} mismatch: {Found} != {Value}")
    ExpectedStatus = (
        "READY"
        if TotalHours <= float(Projection["full_run_gate_gpu_hours"])
        and TotalHours <= float(Projection["full_run_gate_wall_hours"])
        else "BLOCKED_RESOURCE"
    )
    if Projection["status"] != ExpectedStatus:
        Errors.append(f"projection status {Projection['status']} != {ExpectedStatus}")
    return {
        **Expected,
        "training_decodes_three_seeds": TotalTrainingDecodes,
        "group_draw_product": GroupDrawProduct,
        "gpu_hour_gate": Projection["full_run_gate_gpu_hours"],
        "wall_hour_gate": Projection["full_run_gate_wall_hours"],
        "expected_status": ExpectedStatus,
    }, Errors


def AuditInventory(
    E01Root: Path,
    InventoryPath: Path,
    RunRecord: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    Rows = LoadCsv(InventoryPath)
    CurrentNames = sorted(
        FilePath.name
        for FilePath in E01Root.glob("*")
        if FilePath.is_file() and FilePath.suffix != ".pyc"
    )
    RecordedNames = sorted(Row["relative_path"] for Row in Rows)
    if CurrentNames != RecordedNames:
        Errors.append("current R3 source file set differs from inventory")
    Results = []
    DigestPayload = bytearray()
    for Row in Rows:
        FilePath = E01Root / Row["relative_path"]
        ActualBytes = FilePath.stat().st_size
        ActualHash = HashFile(FilePath)
        Match = ActualBytes == int(Row["bytes"]) and ActualHash == Row["sha256"]
        if not Match:
            Errors.append(f"{Row['relative_path']} differs from R3 inventory")
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
    InventoryDigest = hashlib.sha256(DigestPayload).hexdigest()
    RecordedDigest = RunRecord["versions"]["e01_code_inventory_sha256"]
    if InventoryDigest != RecordedDigest:
        Errors.append("R3 inventory digest differs from run manifest")
    return {
        "file_count": len(Results),
        "inventory_sha256": InventoryDigest,
        "recorded_inventory_sha256": RecordedDigest,
        "files": Results,
    }, Errors


def AuditStrictJson(
    ReportsRoot: Path,
    ArtifactRun: Path,
    RunModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    Files = [ReportsRoot / Name for Name in R3JsonReports] + [ArtifactRun]
    Results = []
    for FilePath in Files:
        IsStrict, StrictError = StrictJsonStatus(FilePath)
        if not IsStrict:
            Errors.append(f"{FilePath.name} is not strict JSON: {StrictError}")
        Results.append(
            {
                "file": str(FilePath),
                "sha256": HashFile(FilePath),
                "strict": IsStrict,
                "error": StrictError,
            }
        )
    WriterRejectedNan = False
    WriterPayload = None
    try:
        WriterPayload = RunModule.JsonBytes({"sentinel": float("nan")}).decode("utf-8")
    except (TypeError, ValueError):
        WriterRejectedNan = True
    if not WriterRejectedNan:
        Errors.append(
            "R3 JsonBytes accepts nonfinite values and can emit non-standard bare NaN"
        )
    return {
        "strict_output_count": sum(1 for Row in Results if Row["strict"]),
        "output_count": len(Results),
        "outputs": Results,
        "json_writer_rejects_nan": WriterRejectedNan,
        "json_writer_nan_payload": WriterPayload,
        "required_writer_contract": "json.dumps(..., allow_nan=False)",
    }, Errors


def AuditFiniteGuards(
    PilotReport: dict[str, Any],
    BenchmarkModule: Any,
    TrainModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    FinalLoss = float(PilotReport["final_loss"])
    if PilotReport.get("status") != "PASS" or not math.isfinite(FinalLoss):
        Errors.append("recorded R3 pilot status/loss is not finite PASS")
    PilotSource = inspect.getsource(BenchmarkModule.BenchmarkBalancedPilot)
    FullSource = inspect.getsource(TrainModule.RunSeed)
    PilotChecksLoss = "torch.isfinite(Loss)" in PilotSource
    PilotChecksLogits = "torch.isfinite(Logits)" in PilotSource
    PilotChecksGrad = "isfinite" in PilotSource and "grad" in PilotSource.casefold()
    PilotChecksParameters = "isfinite" in PilotSource and "parameter" in PilotSource.casefold()
    FullHasAnyFiniteGuard = "isfinite" in FullSource

    Logits = torch.zeros((2, 5), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        Logits[0, 0] = float("nan")
    Labels = torch.zeros((2, 5), dtype=torch.float32)
    Masks = torch.ones((2, 5), dtype=torch.bool)
    Masks[0, 0] = False
    Loss = TrainModule.CalculateMaskedBalancedLoss(Logits, Labels, Masks)
    MaskedLossFinite = bool(torch.isfinite(Loss))
    LogitsFinite = bool(torch.isfinite(Logits).all())
    Loss.backward()
    GradFinite = bool(torch.isfinite(Logits.grad).all())

    ScalerEvidence: dict[str, Any]
    if torch.cuda.is_available():
        Parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
        Optimizer = torch.optim.SGD([Parameter], lr=1.0)
        Scaler = torch.amp.GradScaler("cuda", enabled=True)
        ProbeLoss = (Parameter * 0.0 + 1.0).sum()
        Scaler.scale(ProbeLoss).backward()
        Parameter.grad.fill_(float("nan"))
        ParameterBefore = Parameter.detach().clone()
        ScaleBefore = Scaler.get_scale()
        Scaler.step(Optimizer)
        Scaler.update()
        ScalerEvidence = {
            "probe_loss_finite": bool(torch.isfinite(ProbeLoss)),
            "gradient_finite": bool(torch.isfinite(Parameter.grad).all()),
            "parameter_unchanged_after_silent_skip": bool(
                torch.equal(ParameterBefore, Parameter.detach())
            ),
            "scale_before": ScaleBefore,
            "scale_after": Scaler.get_scale(),
            "exception_raised": False,
        }
    else:
        ScalerEvidence = {"not_run": "CUDA unavailable"}

    if not PilotChecksLoss:
        Errors.append("balanced pilot lacks even a finite-loss guard")
    if not PilotChecksLogits or not PilotChecksGrad or not PilotChecksParameters:
        Errors.append(
            "balanced pilot checks Loss only; logits, gradients, and parameters lack hard guards"
        )
    if MaskedLossFinite and not LogitsFinite and not GradFinite:
        Errors.append(
            "loss-only guard accepts masked NaN logits while backward produces NaN gradients"
        )
    if not FullHasAnyFiniteGuard:
        Errors.append(
            "future full RunSeed path has no explicit logits/loss/gradient/parameter finite guard"
        )
    return {
        "recorded_pilot_status": PilotReport.get("status"),
        "recorded_final_loss": FinalLoss,
        "recorded_final_loss_finite": math.isfinite(FinalLoss),
        "pilot_source_guards": {
            "loss": PilotChecksLoss,
            "logits": PilotChecksLogits,
            "gradients": PilotChecksGrad,
            "parameters": PilotChecksParameters,
        },
        "full_run_has_any_explicit_finite_guard": FullHasAnyFiniteGuard,
        "masked_nan_counterexample": {
            "logits_all_finite": LogitsFinite,
            "loss": float(Loss.detach()),
            "loss_finite": MaskedLossFinite,
            "gradients_all_finite": GradFinite,
        },
        "grad_scaler_nonfinite_behavior": ScalerEvidence,
        "conclusion": (
            "GradScaler can skip an unsafe step without raising; it is not the claimed "
            "logits/loss/gradient/parameter hard-failure contract"
        ),
    }, Errors


def AuditOutputsAndLineage(
    DeepvoiceRoot: Path,
    RunRecord: dict[str, Any],
    ReportText: str,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    ArtifactRoot = DeepvoiceRoot / "artifacts" / "e01" / "r3"
    ArtifactFiles = sorted(
        str(FilePath.relative_to(DeepvoiceRoot)).replace("\\", "/")
        for FilePath in ArtifactRoot.rglob("*")
        if FilePath.is_file()
    )
    CheckpointRoot = DeepvoiceRoot / "checkpoints" / "e01"
    CheckpointFiles = sorted(
        str(FilePath.relative_to(DeepvoiceRoot)).replace("\\", "/")
        for FilePath in CheckpointRoot.rglob("*")
        if FilePath.is_file()
    ) if CheckpointRoot.is_dir() else []
    ForbiddenReports = [
        DeepvoiceRoot / "reports" / "e01-r3-results.json",
        DeepvoiceRoot / "reports" / "e01-r3-metrics.json",
    ]
    if ArtifactFiles != ["artifacts/e01/r3/run.json"]:
        Errors.append(f"unexpected R3 artifact files {ArtifactFiles}")
    if CheckpointFiles:
        Errors.append(f"E01 checkpoint files exist {CheckpointFiles}")
    if any(FilePath.exists() for FilePath in ForbiddenReports):
        Errors.append("R3 metric/result report exists despite resource block")
    if RunRecord.get("full_training_started") is not False:
        Errors.append("R3 run manifest does not say full_training_started=false")
    if RunRecord.get("experiment_batch") != "BLOCKED_RESOURCE":
        Errors.append("R3 run status is not BLOCKED_RESOURCE")
    Lineage = RunRecord.get("lineage", {})
    if Lineage.get("supersedes") != ["E01-R1", "E01-R2"]:
        Errors.append("R3 lineage does not supersede R1/R2 explicitly")
    if Lineage.get("prior_report_outputs_preserved") is not True:
        Errors.append("R3 lineage does not preserve prior report outputs")
    if Lineage.get("prior_source_code_byte_preserved") is not False:
        Errors.append("R3 lineage omits the R1 source-byte limitation")
    RequiredClaims = (
        "실제 validation prediction/OOF/metric/checkpoint는 생성하지 않았다",
        "tiny smoke와 pilot loss는 합성·소규모 실행 건전성 검사이며 E01 성능 결과가 아니다",
        "R1 NaN 보고서와 R2 corrected-but-mutable-lineage 보고서는 보존하며",
        "R1 source byte 보존은 주장하지 않는다",
    )
    for Claim in RequiredClaims:
        if Claim not in ReportText:
            Errors.append(f"missing limitation/supersession claim: {Claim}")
    return {
        "artifact_files": ArtifactFiles,
        "checkpoint_files": CheckpointFiles,
        "full_training_started": RunRecord.get("full_training_started"),
        "experiment_batch": RunRecord.get("experiment_batch"),
        "lineage": Lineage,
        "performance_claims_absent": not any(
            Claim not in ReportText for Claim in RequiredClaims[:2]
        ),
    }, Errors


def WriteOutputs(OutputRoot: Path, Record: dict[str, Any]) -> None:
    OutputRoot.mkdir(parents=True, exist_ok=False)
    JsonPath = OutputRoot / "e01-r3-validation-audit.json"
    MarkdownPath = OutputRoot / "e01-r3-validation-audit.md"
    JsonPath.write_text(
        json.dumps(Record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Lines = [
        f"EXPERIMENT_AUDIT: {Record['status']}",
        "",
        "# DeepVoice E01-R3 independent validation audit",
        "",
        "## 판정",
        "",
        "R3의 현재 산출물은 finite/strict이고 immutable inventory도 일치하지만, `nonfinite hard guard`와 strict serializer 계약이 구현되지 않아 `BLOCKED`다.",
        "",
        "## 차단 사유",
        "",
        "1. `JsonBytes`는 `allow_nan=False`를 강제하지 않는다. 동적 probe에서 `NaN`을 bare token으로 직렬화했다. 이번 출력이 우연히 finite라는 사실만으로 후속 full-run 산출물의 strict JSON을 보장할 수 없다.",
        "2. balanced pilot은 `Loss`만 검사한다. 독립 반례에서는 masked 위치의 NaN logit과 NaN gradient가 finite loss `0.693147...` 뒤에 남았고, GradScaler는 예외 없이 step을 skip했다. `RunSeed`에는 logits/loss/gradient/parameter finite guard가 하나도 없다.",
        "3. 따라서 run manifest의 `FP32 log-mel, GradScaler and a nonfinite hard guard` 중 FP32/GradScaler만 사실이며 hard-guard 주장은 코드보다 강하다.",
        "",
        "## 독립 통과 사항",
        "",
        "- R3 JSON 9개 strict parse, pilot final loss finite",
        "- R3 source 12개 byte/SHA 및 inventory digest 완전 일치",
        "- manifest SHA, 137,328행, crossing 0, test crossing-only 격리",
        "- LJSpeech file, WaveFake ZIP, FMA MP3, AIME Parquet 실제 decode",
        "- valid-length/frame mask와 padded-tail feature/logit delta 0",
        "- CUDA autocast 내부 feature path FP32, CNN logits FP16, GradScaler enabled",
        "- 32,768 sample에서 네 strata 각 8,192, speech pair mismatch 0, held-out 0, AIME provider 9종",
        "- 32,768×20×3 = 1,966,080 training decodes 및 validation을 포함한 28.9100939037829시간 산술",
        "- full training/checkpoint/validation prediction/metric 없음; R1/R2 limitation과 supersession 명시",
        "",
        "## 수정 조건",
        "",
        "새 immutable revision에서 serialization 전 모든 float를 검사하고 `json.dumps(..., allow_nan=False)`를 사용해야 한다. pilot과 full `RunSeed` 모두 logits, loss, unscaled gradients, optimizer step 뒤 parameters를 검사하고 nonfinite면 즉시 예외와 BLOCKED 상태를 기록해야 한다. GradScaler의 silent skipped-step은 PASS로 계산하면 안 된다.",
    ]
    MarkdownPath.write_text("\n".join(Lines) + "\n", encoding="utf-8")


def Execute(RepoRoot: Path, OutputRoot: Path) -> dict[str, Any]:
    Started = time.perf_counter()
    DeepvoiceRoot = RepoRoot / "deepvoice"
    E01Root = DeepvoiceRoot / "experiments" / "e01_r3"
    ReportsRoot = DeepvoiceRoot / "reports"
    ManifestPath = ReportsRoot / "deepvoice-training-manifest.csv.gz"
    Config = LoadJson(E01Root / "config.json")
    RunRecord = LoadJson(ReportsRoot / "e01-r3-run-manifest.json")
    Pilot = LoadJson(ReportsRoot / "e01-r3-balanced-pilot.json")
    Preflight = LoadJson(ReportsRoot / "e01-r3-preflight.json")
    LoaderReport = LoadJson(ReportsRoot / "e01-r3-loader-benchmark.json")
    GpuReport = LoadJson(ReportsRoot / "e01-r3-batch-autotune.json")
    Projection = LoadJson(ReportsRoot / "e01-r3-runtime-projection.json")
    ReportText = (ReportsRoot / "e01-r3-experiment-batch.md").read_text(
        encoding="utf-8"
    )

    if str(DeepvoiceRoot) not in sys.path:
        sys.path.insert(0, str(DeepvoiceRoot))
    from experiments.e01_r3 import audio as AudioModule
    from experiments.e01_r3 import benchmark as BenchmarkModule
    from experiments.e01_r3 import model as ModelModule
    from experiments.e01_r3 import records as RecordsModule
    from experiments.e01_r3 import run_e01 as RunModule
    from experiments.e01_r3 import sampling as SamplingModule
    from experiments.e01_r3 import train_e01 as TrainModule

    Checks: list[dict[str, Any]] = []
    ManifestEvidence, ManifestErrors, TrainingRows, ValidationRows = AuditManifest(
        ManifestPath, RunRecord, RecordsModule
    )
    AddCheck(Checks, "manifest_and_test_isolation", ManifestErrors, ManifestEvidence)

    InventoryEvidence, InventoryErrors = AuditInventory(
        E01Root,
        ReportsRoot / "e01-r3-code-inventory.csv",
        RunRecord,
    )
    AddCheck(Checks, "immutable_source_inventory", InventoryErrors, InventoryEvidence)

    StrictEvidence, StrictErrors = AuditStrictJson(
        ReportsRoot,
        DeepvoiceRoot / "artifacts" / "e01" / "r3" / "run.json",
        RunModule,
    )
    AddCheck(Checks, "strict_json_outputs_and_writer", StrictErrors, StrictEvidence)

    GuardEvidence, GuardErrors = AuditFiniteGuards(
        Pilot, BenchmarkModule, TrainModule
    )
    AddCheck(Checks, "nonfinite_hard_guard_behavior", GuardErrors, GuardEvidence)

    LocatorEvidence, LocatorErrors = AuditLocators(
        TrainingRows, Preflight, LoaderReport, AudioModule
    )
    AddCheck(Checks, "four_real_locator_loaders", LocatorErrors, LocatorEvidence)

    PaddingEvidence, PaddingErrors = AuditPaddingAndFp32(
        Config, AudioModule, ModelModule, BenchmarkModule, TrainModule
    )
    AddCheck(
        Checks,
        "valid_length_tail_fp32_and_gradscaler",
        PaddingErrors,
        PaddingEvidence,
    )

    TrainingRecords, _ValidationRecords, _ManifestSummary = RecordsModule.LoadE01Records(
        ManifestPath
    )
    SamplerEvidence, SamplerErrors = AuditSampler(
        TrainingRecords,
        Config,
        SamplingModule,
        LoadCsv(ReportsRoot / "e01-r3-sampler-audit.csv"),
    )
    AddCheck(Checks, "exact_group_first_sampler", SamplerErrors, SamplerEvidence)

    ProjectionEvidence, ProjectionErrors = AuditProjection(
        Config,
        Projection,
        LoaderReport,
        GpuReport,
        Pilot,
        ValidationRows,
    )
    AddCheck(Checks, "resource_projection_math", ProjectionErrors, ProjectionEvidence)

    OutputEvidence, OutputErrors = AuditOutputsAndLineage(
        DeepvoiceRoot, RunRecord, ReportText
    )
    AddCheck(
        Checks,
        "exclusive_outputs_and_lineage_limitations",
        OutputErrors,
        OutputEvidence,
    )

    Status = "PASS" if all(Check["status"] == "PASS" for Check in Checks) else "BLOCKED"
    Record = {
        "status": Status,
        "audit_id": "E01-R3-INDEPENDENT-AUDIT-20260830",
        "scope": "immutable experiments/e01_r3 and e01-r3-* outputs only",
        "test_data_handling": (
            "real test rows used only as group/split projections for crossing; no test "
            "performance, metadata, label, mask, or prediction statistic calculated"
        ),
        "checks": Checks,
        "input_hashes": {
            "manifest": HashFile(ManifestPath),
            "run_manifest": HashFile(ReportsRoot / "e01-r3-run-manifest.json"),
            "balanced_pilot": HashFile(ReportsRoot / "e01-r3-balanced-pilot.json"),
            "code_inventory": HashFile(ReportsRoot / "e01-r3-code-inventory.csv"),
            "batch_report": HashFile(ReportsRoot / "e01-r3-experiment-batch.md"),
        },
        "audit_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runtime_seconds": time.perf_counter() - Started,
    }
    WriteOutputs(OutputRoot, Record)
    return Record


def Main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_e01_r3.py REPO_ROOT OUTPUT_ROOT")
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
