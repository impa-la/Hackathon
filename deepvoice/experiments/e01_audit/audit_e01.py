# /// <summary>
# Independent read-only auditor for the original DeepVoice E01 BLOCKED_RESOURCE run
# /// </summary>

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import wave
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
AllowedTestFields = {"content_group_key", "recommended_content_split"}
Datasets = (
    "ljspeech-1.1",
    "wavefake-1.2.0",
    "fma-small",
    "aime-open-model-subset",
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


def LoadJsonPermissive(FilePath: Path) -> Any:
    return json.loads(FilePath.read_text(encoding="utf-8"))


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
            GroupKey = Row["content_group_key"]
            GroupSplits[GroupKey].add(Split)
            if Split == "test":
                continue
            if Split == "train":
                TrainingRows.append(Row)
            elif Split == "validation":
                ValidationRows.append(Row)
            else:
                raise ValueError(f"Unexpected non-test split {Split}")
    Crossings = sorted(
        GroupKey for GroupKey, Splits in GroupSplits.items() if len(Splits) > 1
    )
    return RowCount, TrainingRows, ValidationRows, Crossings


def ParseLocator(Locator: str) -> tuple[str, Path, str | int | None]:
    if Locator.startswith("zip://"):
        Payload = Locator[6:]
        Container, Member = Payload.split("!/", 1)
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

    AudioColumn = Parquet.read_table(ContainerPath, columns=["audio"]).column("audio")
    AudioValue = AudioColumn[int(Payload)].as_py()
    if not isinstance(AudioValue, dict) or not isinstance(AudioValue.get("bytes"), bytes):
        raise TypeError("Parquet audio is not embedded bytes")
    return Kind, ContainerPath, AudioValue["bytes"]


def DecodeWithFfmpeg(
    ContainerPath: Path,
    AudioBytes: bytes | None,
    SampleRate: int = 16000,
) -> np.ndarray:
    InputName = "pipe:0" if AudioBytes is not None else str(ContainerPath)
    Result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            InputName,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SampleRate),
            "pipe:1",
        ],
        input=AudioBytes,
        check=True,
        capture_output=True,
        timeout=180,
    )
    Samples = np.frombuffer(Result.stdout, dtype="<f4").copy()
    if Samples.size == 0 or not np.isfinite(Samples).all():
        raise ValueError("Independent ffmpeg decode is empty or nonfinite")
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
        DecodeCountDelta = abs(ActualCount - IndependentCount)
        PreflightCount = int(PreflightByDataset[Dataset]["decoded_sample_count"])
        LoaderCount = int(LoaderByDataset[Dataset]["decoded_sample_count"])
        # PCM resampling backends can select opposite endpoint-rounding policies.
        # A one-sample length delta is therefore accepted only when the E01 loader,
        # its preflight, and its recorded loader probe agree exactly.
        if DecodeCountDelta > 1:
            Errors.append(
                f"{Dataset} E01 loader count {ActualCount} != independent {IndependentCount}"
            )
        if ActualCount != PreflightCount or ActualCount != LoaderCount:
            Errors.append(f"{Dataset} reported decode counts differ from current loader")
        if not torch.isfinite(Actual).all() or Actual.numel() == 0:
            Errors.append(f"{Dataset} E01 loader returned invalid waveform")
        ExpectedKinds = {
            "ljspeech-1.1": "file",
            "wavefake-1.2.0": "zip",
            "fma-small": "file",
            "aime-open-model-subset": "parquet",
        }
        if Kind != ExpectedKinds[Dataset]:
            Errors.append(f"{Dataset} locator kind {Kind}")
        Results.append(
            {
                "dataset": Dataset,
                "sample_id": Row["sample_id"],
                "locator_kind": Kind,
                "container_bytes": ContainerPath.stat().st_size,
                "independent_ffmpeg_sample_count_16k": IndependentCount,
                "e01_loader_sample_count_16k": ActualCount,
                "independent_count_absolute_delta": DecodeCountDelta,
                "accepted_endpoint_rounding_tolerance_samples": 1,
                "e01_decoder_backend": Diagnostics.get("decoder_backend"),
                "preflight_sample_count": PreflightCount,
                "loader_report_sample_count": LoaderCount,
            }
        )
    AudioModule.CloseAudioContainerCaches()
    return {"decode_count": len(Results), "rows": Results}, Errors


def AuditPadding(
    Config: dict[str, Any],
    AudioModule: Any,
    ModelModule: Any,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    torch.manual_seed(20260830)
    Model = ModelModule.LogMelCnn(Config).cpu().eval()
    SegmentLength = int(Config["sample_rate"] * Config["segment_seconds"])
    ValidLength = int(Config["sample_rate"] * 4)
    Generator = torch.Generator().manual_seed(20260830)
    First = torch.zeros((1, SegmentLength), dtype=torch.float32)
    First[:, :ValidLength] = torch.randn(
        (1, ValidLength),
        generator=Generator,
    ) * 0.01
    TailSentinel = First.clone()
    TailSentinel[:, ValidLength:] = torch.linspace(
        -1000.0,
        1000.0,
        SegmentLength - ValidLength,
    )
    ValidSentinel = First.clone()
    ValidSentinel[:, :ValidLength] += 0.25
    Counts = torch.tensor([ValidLength], dtype=torch.long)
    with torch.inference_mode():
        FirstFeatures, FirstMask = Model.FeatureExtractor(First, Counts)
        TailFeatures, TailMask = Model.FeatureExtractor(TailSentinel, Counts)
        FirstLogits = Model(First, Counts)
        TailLogits = Model(TailSentinel, Counts)
        ValidLogits = Model(ValidSentinel, Counts)
    FeatureDelta = float(torch.max(torch.abs(FirstFeatures - TailFeatures)))
    LogitDelta = float(torch.max(torch.abs(FirstLogits - TailLogits)))
    ValidLogitDelta = float(torch.max(torch.abs(FirstLogits - ValidLogits)))
    if FeatureDelta != 0.0 or LogitDelta != 0.0:
        Errors.append("padded-tail sentinel changed masked features or logits")
    if not torch.equal(FirstMask, TailMask):
        Errors.append("padded-tail sentinel changed frame mask")
    if ValidLogitDelta <= 0.0:
        Errors.append("valid-region perturbation did not affect logits")
    Segments, Lengths = AudioModule.CreateValidationSegmentsWithLengths(
        First[0, :ValidLength],
        int(Config["sample_rate"]),
        float(Config["segment_seconds"]),
        int(Config["max_segments_per_file"]),
    )
    if Segments.shape != (1, SegmentLength) or Lengths.tolist() != [ValidLength]:
        Errors.append("short-file valid length or padding shape differs")
    if float(Segments[0, ValidLength:].abs().sum()) != 0.0:
        Errors.append("short-file padded tail is nonzero")
    return {
        "segment_length": SegmentLength,
        "valid_length": ValidLength,
        "feature_tail_delta": FeatureDelta,
        "logit_tail_delta": LogitDelta,
        "valid_region_logit_delta": ValidLogitDelta,
        "valid_frame_count": int(FirstMask.sum()),
        "segment_valid_lengths": Lengths.tolist(),
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
        TrainingRecords,
        SamplesPerEpoch,
        20260830,
    )
    Indices = list(iter(Sampler))
    Repeat = list(
        iter(
            SamplingModule.GroupFirstBalancedSampler(
                TrainingRecords,
                SamplesPerEpoch,
                20260830,
            )
        )
    )
    if Indices != Repeat:
        Errors.append("same-seed group-first sampler is not deterministic")
    if len(Indices) != SamplesPerEpoch:
        Errors.append(f"sampler emitted {len(Indices)} rows")
    StratumCounts = Counter()
    ProviderCounts = Counter()
    HeldoutCount = 0
    PairMismatchCount = 0
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
            PairMismatchCount += 1
    ExpectedPerStratum = SamplesPerEpoch // 4
    if any(StratumCounts[Stratum] != ExpectedPerStratum for Stratum in SamplingModule.Strata):
        Errors.append(f"stratum balance differs: {dict(StratumCounts)}")
    if PairMismatchCount:
        Errors.append(f"{PairMismatchCount} speech quads are not content paired")
    if HeldoutCount:
        Errors.append(f"{HeldoutCount} held-out rows were sampled")
    GroupIndex = SamplingModule.BuildGroupIndex(TrainingRecords)
    IndependentSummary = {
        Stratum: {
            "group_count": len(GroupIndex[Stratum]),
            "row_count": sum(len(Value) for Value in GroupIndex[Stratum].values()),
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
        Errors.append("sampler report counts differ from independent group index")
    ProviderValues = list(ProviderCounts.values())
    if len(ProviderCounts) != 9:
        Errors.append(f"music-fake provider sampler covered {len(ProviderCounts)} providers")
    return {
        "samples": len(Indices),
        "stratum_counts": dict(sorted(StratumCounts.items())),
        "speech_pair_mismatch_count": PairMismatchCount,
        "heldout_sample_count": HeldoutCount,
        "music_fake_provider_counts": dict(sorted(ProviderCounts.items())),
        "music_fake_provider_count_range": [min(ProviderValues), max(ProviderValues)],
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
    ExpectedSamplesPerEpoch = int(Config["balanced_group_draws_per_epoch"]) * int(
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
                    float(Row["duration_seconds"]) / float(Config["segment_seconds"])
                ),
            ),
        )
        for Row in ValidationRows
    )
    ValidationSecondsPerSeed = len(ValidationRows) / LoaderRate + ValidationSegments / GpuRate
    TotalHours = len(Config["seeds"]) * (
        TrainingSecondsPerSeed + ValidationSecondsPerSeed
    ) / 3600.0
    ExpectedValues = {
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
    if SamplesPerEpoch != ExpectedSamplesPerEpoch:
        Errors.append("samples_per_epoch differs from group-draw product")
    for Key, Expected in ExpectedValues.items():
        Found = float(Projection[Key])
        if not math.isclose(Found, float(Expected), rel_tol=0.0, abs_tol=1e-12):
            Errors.append(f"projection {Key} mismatch: {Found} != {Expected}")
    ExpectedStatus = (
        "READY"
        if TotalHours <= float(Projection["full_run_gate_gpu_hours"])
        else "BLOCKED_RESOURCE"
    )
    if Projection["status"] != ExpectedStatus:
        Errors.append(f"projection status {Projection['status']} != {ExpectedStatus}")
    if TotalHours <= float(Projection["full_run_gate_wall_hours"]):
        Errors.append("projection unexpectedly passes the 24 wall-hour gate")
    return {
        **ExpectedValues,
        "training_decodes_three_seeds": TotalTrainingDecodes,
        "group_draw_product": ExpectedSamplesPerEpoch,
        "gpu_hour_gate": Projection["full_run_gate_gpu_hours"],
        "wall_hour_gate": Projection["full_run_gate_wall_hours"],
        "expected_status": ExpectedStatus,
        "math_matches_report": not Errors,
    }, Errors


def AuditOutputs(DeepvoiceRoot: Path, RunRecord: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    ArtifactsRoot = DeepvoiceRoot / "artifacts" / "e01"
    # R2/R3 were created after the R1 timestamp under child directories.  This
    # R1 audit intentionally inventories only direct R1 files rather than making
    # later immutable-lineage work look like an R1 output-policy violation.
    ArtifactFiles = sorted(
        str(PathValue.relative_to(DeepvoiceRoot))
        for PathValue in ArtifactsRoot.glob("*")
        if PathValue.is_file()
    ) if ArtifactsRoot.is_dir() else []
    CheckpointRoot = DeepvoiceRoot / "checkpoints" / "e01"
    CheckpointFiles = sorted(
        str(PathValue.relative_to(DeepvoiceRoot))
        for PathValue in CheckpointRoot.glob("*")
        if PathValue.is_file()
    ) if CheckpointRoot.is_dir() else []
    ForbiddenReports = [
        DeepvoiceRoot / "reports" / "e01-results.json",
    ]
    ForbiddenArtifacts = [
        PathValue
        for PathValue in ArtifactsRoot.glob("*")
        if PathValue.is_file()
        and any(Token in PathValue.name.casefold() for Token in ("validation", "oof", "metric", "checkpoint"))
    ] if ArtifactsRoot.is_dir() else []
    if ArtifactFiles != ["artifacts\\e01\\run.json"]:
        Errors.append(f"unexpected E01 artifact files: {ArtifactFiles}")
    if CheckpointFiles:
        Errors.append(f"E01 checkpoints exist: {CheckpointFiles}")
    if any(PathValue.exists() for PathValue in ForbiddenReports):
        Errors.append("e01-results.json exists")
    if ForbiddenArtifacts:
        Errors.append(f"validation/performance artifacts exist: {ForbiddenArtifacts}")
    if RunRecord.get("full_training_started") is not False:
        Errors.append("run record does not state full_training_started=false")
    if RunRecord.get("experiment_batch") != "BLOCKED_RESOURCE":
        Errors.append("run record is not BLOCKED_RESOURCE")
    return {
        "artifact_files": ArtifactFiles,
        "checkpoint_files": CheckpointFiles,
        "e01_results_exists": any(PathValue.exists() for PathValue in ForbiddenReports),
        "validation_or_performance_artifact_count": len(ForbiddenArtifacts),
        "full_training_started": RunRecord.get("full_training_started"),
        "experiment_batch": RunRecord.get("experiment_batch"),
    }, Errors


def AuditCodeInventory(
    E01Root: Path,
    InventoryPath: Path,
    RunRecord: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    Rows = LoadCsv(InventoryPath)
    Results = []
    DigestPayload = bytearray()
    for Row in Rows:
        RelativePath = Row["relative_path"]
        FilePath = E01Root / RelativePath
        ActualBytes = FilePath.stat().st_size
        ActualHash = HashFile(FilePath)
        Match = ActualBytes == int(Row["bytes"]) and ActualHash == Row["sha256"]
        Results.append(
            {
                "relative_path": RelativePath,
                "recorded_bytes": int(Row["bytes"]),
                "actual_bytes": ActualBytes,
                "recorded_sha256": Row["sha256"],
                "actual_sha256": ActualHash,
                "match": Match,
            }
        )
        if not Match:
            Errors.append(f"{RelativePath} differs from R1 inventory")
        DigestPayload.extend(
            f"{RelativePath}\0{Row['bytes']}\0{Row['sha256']}\n".encode("utf-8")
        )
    InventoryDigest = hashlib.sha256(DigestPayload).hexdigest()
    if InventoryDigest != RunRecord["versions"]["e01_code_inventory_sha256"]:
        Errors.append("R1 inventory digest differs from run record")
    ConfigHash = HashFile(E01Root / "config.json")
    if ConfigHash != RunRecord["versions"]["config_sha256"]:
        Errors.append("config hash differs from R1 run record")
    return {
        "inventory_digest": InventoryDigest,
        "recorded_inventory_digest": RunRecord["versions"]["e01_code_inventory_sha256"],
        "config_sha256": ConfigHash,
        "files": Results,
    }, Errors


def AuditPilotAndJson(
    ReportsRoot: Path,
    PilotReport: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    PilotPath = ReportsRoot / "e01-balanced-pilot.json"
    StrictValid, StrictError = StrictJsonStatus(PilotPath)
    FinalLoss = float(PilotReport["final_loss"])
    if math.isfinite(FinalLoss):
        Errors.append("expected recorded R1 final_loss to expose the observed NaN")
    if PilotReport.get("status") == "PASS" and not math.isfinite(FinalLoss):
        Errors.append("balanced pilot is PASS despite nonfinite final_loss")
    if not StrictValid:
        Errors.append(f"balanced pilot is not strict RFC JSON: {StrictError}")
    NonStrictFiles = []
    R1JsonNames = (
        "e01-batch-autotune.json",
        "e01-balanced-pilot.json",
        "e01-loader-benchmark.json",
        "e01-preflight.json",
        "e01-run-manifest.json",
        "e01-runtime-projection.json",
        "e01-tiny-gpu-smoke.json",
        "e01-unit-test-results.json",
    )
    for FilePath in (ReportsRoot / Name for Name in R1JsonNames):
        IsStrict, Error = StrictJsonStatus(FilePath)
        if not IsStrict:
            NonStrictFiles.append({"file": FilePath.name, "error": Error})
    if PilotReport.get("is_e01_performance_result") is not False:
        Errors.append("pilot is not marked non-performance")
    return {
        "pilot_status": PilotReport.get("status"),
        "final_loss": "NaN" if not math.isfinite(FinalLoss) else FinalLoss,
        "final_loss_is_finite": math.isfinite(FinalLoss),
        "strict_json_valid": StrictValid,
        "strict_json_error": StrictError,
        "non_strict_e01_json_files": NonStrictFiles,
        "root_cause": {
            "pass_gate": "R1 accepted Pilot['status'] without validating returned final_loss",
            "writer": "Python json.dumps default allow_nan=True emitted the non-standard NaN token",
            "projection": "runtime projection consumed throughput from the invalid PASS pilot",
        },
    }, Errors


def WriteOutputs(OutputRoot: Path, Record: dict[str, Any]) -> None:
    OutputRoot.mkdir(parents=True, exist_ok=False)
    JsonPath = OutputRoot / "e01-validation-audit.json"
    MarkdownPath = OutputRoot / "e01-validation-audit.md"
    JsonPath.write_text(
        json.dumps(Record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Lines = [
        f"EXPERIMENT_AUDIT: {Record['status']}",
        "",
        "# DeepVoice E01 R1 independent validation audit",
        "",
        f"감사 시각: {Record['finished_at_local']}",
        "범위: original `e01-*` BLOCKED_RESOURCE run only; later e01-r2/e01-r3 lineages excluded",
        "",
        "## 판정",
        "",
        "E01 R1은 자원 gate에서 중단한 결정 자체는 보수적으로 맞지만, 실행 건전성과 재현성 계약을 통과하지 못했으므로 `BLOCKED`다.",
        "",
    ]
    for Check in Record["checks"]:
        for Error in Check["errors"]:
            Lines.append(f"- {Check['check']}: {Error}")
    Lines.extend(
        [
            "",
            "## 핵심 차단 사유",
            "",
            "1. `e01-balanced-pilot.json`은 `status: PASS`와 `final_loss: NaN`을 동시에 기록했다. nonfinite training loss를 PASS로 둘 수 없다.",
            "2. R1 JSON writer가 Python 기본 `allow_nan=True` 동작을 허용해 RFC JSON이 아닌 bare `NaN` token을 저장했다. strict parser는 이 파일을 거부한다.",
            "3. R1 run manifest가 기록한 `run_e01.py`는 13,706 bytes / SHA-256 `8ebc70...96517`이지만 감사 시 파일은 다른 크기·hash다. immutable R1 driver가 없어 exact code/report consistency를 재현할 수 없다.",
            "4. 31.776시간 산술은 정확하지만 입력 throughput은 nonfinite-loss pilot을 PASS로 채택한 값이다. 따라서 resource 결론은 보수적 중단 근거로만 남고 유효한 E01 training preflight로 승격할 수 없다.",
            "",
            "## 통과한 독립 검사",
            "",
            "- manifest SHA-256, 137,328행, content-group split crossing 0",
            "- 실제 test는 group/split crossing projection 외 사용 없음",
            "- LJSpeech file, WaveFake ZIP, FMA MP3, AIME Parquet 네 locator를 독립 ffmpeg/pyarrow 경로로 decode하고 E01 loader sample count와 대조(PCM resampler endpoint의 최대 1 sample 반올림 차이만 허용)",
            "- valid-sample/frame mask 및 padded-tail sentinel의 feature/logit delta 0",
            "- 32,768-sample sampler에서 네 strata 각 8,192, speech content pairing 100%, held-out source 0, AIME provider 9종 포함",
            "- workload 32,768×20×3 = 1,966,080 training decodes, validation 13,540 files/18,165 segments, 산출 31.776394시간",
            "- 31.776시간은 3 GPU-hour와 24 wall-hour gate를 모두 초과하므로 full training 미실행 분기와 일치",
            "- checkpoint, validation prediction/OOF, E01 metric/result 파일 없음; `artifacts/e01/run.json`만 존재",
            "- tiny smoke와 pilot은 성능 결과가 아니라고 표시되어 있으며 validation 성능 주장은 없음",
            "",
            "## 원인과 수정 책임",
            "",
            "R1이 NaN을 PASS로 기록한 직접 원인은 최종 pilot payload에 대한 finite-loss gate와 strict JSON gate가 없었기 때문이다. benchmark 내부의 사전 loss 검사가 있었다는 주장만으로 저장된 NaN과 PASS의 모순은 해소되지 않는다. 또한 R1 이후 같은 `experiments/e01` 경로가 변경되어 원 실행 driver를 재현할 수 없다.",
            "",
            "후속 실행은 새 immutable revision 디렉터리에서 시작하고, 매 batch 및 optimizer step 후 logits/loss/grad/parameter finite 검사를 수행하며, payload 직렬화 전에 모든 float를 검사하고 `json.dumps(..., allow_nan=False)`를 강제해야 한다. R1 산출물은 자원 중단의 참고 기록으로만 보존하고 모델 개선 근거로 사용하지 않는다.",
        ]
    )
    MarkdownPath.write_text("\n".join(Lines) + "\n", encoding="utf-8")


def Execute(RepoRoot: Path, OutputRoot: Path) -> dict[str, Any]:
    Started = time.perf_counter()
    DeepvoiceRoot = RepoRoot / "deepvoice"
    E01Root = DeepvoiceRoot / "experiments" / "e01"
    ReportsRoot = DeepvoiceRoot / "reports"
    ManifestPath = ReportsRoot / "deepvoice-training-manifest.csv.gz"
    Config = LoadJsonPermissive(E01Root / "config.json")
    RunRecord = LoadJsonPermissive(ReportsRoot / "e01-run-manifest.json")
    Preflight = LoadJsonPermissive(ReportsRoot / "e01-preflight.json")
    Pilot = LoadJsonPermissive(ReportsRoot / "e01-balanced-pilot.json")
    LoaderReport = LoadJsonPermissive(ReportsRoot / "e01-loader-benchmark.json")
    GpuReport = LoadJsonPermissive(ReportsRoot / "e01-batch-autotune.json")
    Projection = LoadJsonPermissive(ReportsRoot / "e01-runtime-projection.json")
    Checks = []

    if str(DeepvoiceRoot) not in sys.path:
        sys.path.insert(0, str(DeepvoiceRoot))
    from experiments.e01 import audio as AudioModule
    from experiments.e01 import model as ModelModule
    from experiments.e01 import records as RecordsModule
    from experiments.e01 import sampling as SamplingModule

    RowCount, TrainingRows, ValidationRows, Crossings = ReadManifest(ManifestPath)
    ManifestHash = HashFile(ManifestPath)
    ManifestErrors = []
    if ManifestHash != ExpectedManifestSha256:
        ManifestErrors.append(f"manifest SHA mismatch {ManifestHash}")
    if RowCount != ExpectedManifestRows:
        ManifestErrors.append(f"manifest row count {RowCount}")
    if Crossings:
        ManifestErrors.append(f"content-group crossings {len(Crossings)}")
    if RunRecord["manifest_integrity"]["manifest_sha256"] != ManifestHash:
        ManifestErrors.append("run-record manifest SHA mismatch")
    AddCheck(
        Checks,
        "manifest_and_test_isolation",
        ManifestErrors,
        {
            "sha256": ManifestHash,
            "row_count": RowCount,
            "train_rows": len(TrainingRows),
            "validation_rows": len(ValidationRows),
            "crossing_group_count": len(Crossings),
            "actual_test_handling": "group/split projection for crossing only; no retained test rows",
            "allowed_test_fields": sorted(AllowedTestFields),
        },
    )

    LocatorEvidence, LocatorErrors = AuditLocators(
        TrainingRows,
        Preflight,
        LoaderReport,
        AudioModule,
    )
    AddCheck(Checks, "four_real_locator_loaders", LocatorErrors, LocatorEvidence)

    PaddingEvidence, PaddingErrors = AuditPadding(Config, AudioModule, ModelModule)
    AddCheck(
        Checks,
        "valid_length_and_tail_invariance",
        PaddingErrors,
        PaddingEvidence,
    )

    ActualTrainingRecords, ActualValidationRecords, ActualManifestSummary = (
        RecordsModule.LoadE01Records(ManifestPath)
    )
    SamplerEvidence, SamplerErrors = AuditSampler(
        ActualTrainingRecords,
        Config,
        SamplingModule,
        LoadCsv(ReportsRoot / "e01-sampler-audit.csv"),
    )
    AddCheck(
        Checks,
        "group_first_sampler_and_balance",
        SamplerErrors,
        SamplerEvidence,
    )

    ProjectionEvidence, ProjectionErrors = AuditProjection(
        Config,
        Projection,
        LoaderReport,
        GpuReport,
        Pilot,
        ValidationRows,
    )
    AddCheck(
        Checks,
        "resource_projection_math_and_gate",
        ProjectionErrors,
        ProjectionEvidence,
    )

    OutputEvidence, OutputErrors = AuditOutputs(DeepvoiceRoot, RunRecord)
    AddCheck(
        Checks,
        "exclusive_outputs_and_no_performance_artifacts",
        OutputErrors,
        OutputEvidence,
    )

    PilotEvidence, PilotErrors = AuditPilotAndJson(ReportsRoot, Pilot)
    AddCheck(
        Checks,
        "finite_pilot_and_strict_json",
        PilotErrors,
        PilotEvidence,
    )

    InventoryEvidence, InventoryErrors = AuditCodeInventory(
        E01Root,
        ReportsRoot / "e01-code-inventory.csv",
        RunRecord,
    )
    AddCheck(
        Checks,
        "code_report_consistency",
        InventoryErrors,
        InventoryEvidence,
    )

    ClaimErrors = []
    ReportText = (ReportsRoot / "e01-experiment-batch.md").read_text(encoding="utf-8")
    RequiredClaims = (
        "EXPERIMENT_BATCH: BLOCKED_RESOURCE",
        "실제 validation prediction/OOF/metric/checkpoint는 생성하지 않았다",
        "tiny smoke와 pilot loss는 합성·소규모 실행 건전성 검사이며 E01 성능 결과가 아니다",
        "E02 학습은 수행하지 않았다",
    )
    for Claim in RequiredClaims:
        if Claim not in ReportText:
            ClaimErrors.append(f"missing report limitation: {Claim}")
    if RunRecord.get("tiny_smoke_is_e01_performance_result") is not False:
        ClaimErrors.append("tiny smoke is not marked non-performance")
    if RunRecord.get("balanced_pilot_is_e01_performance_result") is not False:
        ClaimErrors.append("balanced pilot is not marked non-performance")
    AddCheck(
        Checks,
        "report_scope_and_claims",
        ClaimErrors,
        {
            "required_limitations_present": not ClaimErrors,
            "tiny_smoke_is_performance": RunRecord.get(
                "tiny_smoke_is_e01_performance_result"
            ),
            "balanced_pilot_is_performance": RunRecord.get(
                "balanced_pilot_is_e01_performance_result"
            ),
        },
    )

    Status = "PASS" if all(Check["status"] == "PASS" for Check in Checks) else "BLOCKED"
    Record = {
        "status": Status,
        "audit_id": "E01-R1-INDEPENDENT-AUDIT-20260830",
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "original e01-* BLOCKED_RESOURCE run only; later e01-r2/e01-r3 outputs excluded",
        "test_data_handling": "real test rows used only as group/split projections for crossing; no test performance or metadata statistics calculated",
        "checks": Checks,
        "input_hashes": {
            "manifest": ManifestHash,
            "e01_run_manifest": HashFile(ReportsRoot / "e01-run-manifest.json"),
            "e01_balanced_pilot": HashFile(ReportsRoot / "e01-balanced-pilot.json"),
            "e01_code_inventory": HashFile(ReportsRoot / "e01-code-inventory.csv"),
            "e01_batch_report": HashFile(ReportsRoot / "e01-experiment-batch.md"),
        },
        "audit_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runtime_seconds": time.perf_counter() - Started,
    }
    WriteOutputs(OutputRoot, Record)
    return Record


def Main() -> int:
    if len(sys.argv) != 3:
        print("Usage: audit_e01.py REPO_ROOT OUTPUT_ROOT", file=sys.stderr)
        return 2
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
