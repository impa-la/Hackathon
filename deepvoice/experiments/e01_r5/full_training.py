# /// <summary>
# Visible, resumable, guarded FP32 full three-seed E01-R5 training
# /// </summary>

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import VerifyCompleteCache
from .contract_adapter import (
    BootstrapByContentGroup,
    BuildShortcutMetricAudit,
    CalculateCompetitionProxy,
    CalculateHeadMetrics,
)
from .determinism import ConfigureParentCpuThreads, PinDataLoaderWorkerCpuThreads
from .model import LogMelCnn
from .numerical import (
    GuardedFp32OptimizationStep,
    RequireFiniteTensor,
    RequireFp32TrainingMode,
    RequireMaskedLabels,
    RequireModelParametersFinite,
)
from .preflight import RunPreflight
from .records import AudioRecord, LoadE01Records
from .resume import (
    AtomicTorchSave,
    AtomicWriteBytes,
    AtomicWriteJson,
    BuildRunIdentity,
    CompleteSeedSchemaVersion,
    CsvBytes,
    FormatDuration,
    HashFile,
    ProgressMath,
    ResumeSchemaVersion,
    StrictLoadJson,
    UtcNow,
    ValidateRunIdentity,
)
from .sampling import GroupFirstBalancedSampler
from .strict_serialization import AssertFinitePayload, JsonBytes
from .train_e01 import (
    BuildPredictionRows,
    CalculateMaskedBalancedLoss,
    CheckModelSingleton,
    EpochTaggedSampler,
    PredictRecord,
    SetSeed,
    TrainingDataset,
    WriteGzipCsv,
)


def ValidateFixedR5Contract(Config: dict[str, Any]) -> None:
    Expected = {
        "revision": "R5",
        "precision_mode": "fp32_guarded",
        "batch_size": 32,
        "workers": 2,
        "samples_per_epoch": 32768,
        "epochs": 20,
        "seeds": [20260830, 20260831, 20260832],
        "sample_rate": 16000,
        "segment_seconds": 8.0,
        "max_segments_per_file": 8,
        "cache_relative_path": "data/cache/e01_r4",
    }
    Mismatches = {
        Key: {"expected": Value, "observed": Config.get(Key)}
        for Key, Value in Expected.items()
        if Config.get(Key) != Value
    }
    if Mismatches:
        raise RuntimeError(f"E01-R5 fixed full-training contract mismatch: {Mismatches}")
    ExpectedSamples = int(Config["balanced_group_draws_per_epoch"]) * int(
        Config["samples_per_balanced_group_draw"]
    )
    if ExpectedSamples != int(Config["samples_per_epoch"]):
        raise RuntimeError("E01-R5 group-balanced samples-per-epoch contract mismatch")


def ValidationSegmentCount(Record: AudioRecord, Config: dict[str, Any]) -> int:
    return min(
        int(Config["max_segments_per_file"]),
        max(1, math.ceil(Record.DurationSeconds / float(Config["segment_seconds"]))),
    )


def CudaMemory() -> dict[str, float]:
    return {
        "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "cuda_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
    }


def RequireNestedTensorsFinite(Value: Any, PathText: str = "$") -> None:
    if torch.is_tensor(Value):
        if Value.is_floating_point() or Value.is_complex():
            RequireFiniteTensor(Value, "resume_checkpoint", -1, PathText)
        return
    if isinstance(Value, dict):
        for Key, Child in Value.items():
            RequireNestedTensorsFinite(Child, f"{PathText}.{Key}")
        return
    if isinstance(Value, (list, tuple)):
        for Index, Child in enumerate(Value):
            RequireNestedTensorsFinite(Child, f"{PathText}[{Index}]")


def AtomicWriteGzipCsv(OutputPath: Path, Rows: Sequence[dict[str, Any]]) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    TemporaryPath = OutputPath.with_name(f"{OutputPath.name}.tmp-{os.getpid()}")
    WriteGzipCsv(TemporaryPath, Rows)
    with TemporaryPath.open("rb+") as FileHandle:
        FileHandle.flush()
        os.fsync(FileHandle.fileno())
    os.replace(TemporaryPath, OutputPath)


def ResumePaths(ArtifactsRoot: Path, Seed: int) -> tuple[Path, Path]:
    ResumeRoot = ArtifactsRoot / "resume"
    return ResumeRoot, ResumeRoot / f"seed-{Seed}-latest.json"


def SaveEpochResume(
    ArtifactsRoot: Path,
    Seed: int,
    NextEpoch: int,
    EpochRows: list[dict[str, Any]],
    NumericalSummary: dict[str, Any],
    Model: LogMelCnn,
    Optimizer: torch.optim.Optimizer,
    Config: dict[str, Any],
    RunIdentity: dict[str, Any],
) -> dict[str, Any]:
    ResumeRoot, LatestPath = ResumePaths(ArtifactsRoot, Seed)
    CheckpointPath = ResumeRoot / f"seed-{Seed}-after-epoch-{NextEpoch:02d}.pt"
    ModelState = Model.state_dict()
    OptimizerState = Optimizer.state_dict()
    RequireNestedTensorsFinite(ModelState, "$.model_state")
    RequireNestedTensorsFinite(OptimizerState, "$.optimizer_state")
    AssertFinitePayload(EpochRows)
    AssertFinitePayload(NumericalSummary)
    Payload = {
        "schema": ResumeSchemaVersion,
        "experiment_id": "E01",
        "revision": "R5",
        "seed": Seed,
        "next_epoch": NextEpoch,
        "epoch_rows": EpochRows,
        "numerical_summary": NumericalSummary,
        "config": Config,
        "run_identity": RunIdentity,
        "model_state": ModelState,
        "optimizer_state": OptimizerState,
        "saved_utc": UtcNow(),
    }
    AtomicTorchSave(CheckpointPath, Payload)
    Pointer = {
        "schema": "e01-r5-resume-pointer-v1",
        "seed": Seed,
        "next_epoch": NextEpoch,
        "checkpoint_relative_path": str(
            CheckpointPath.relative_to(ArtifactsRoot)
        ).replace("\\", "/"),
        "checkpoint_sha256": HashFile(CheckpointPath),
        "epoch_rows_count": len(EpochRows),
        "run_identity": RunIdentity,
        "saved_utc": UtcNow(),
    }
    AtomicWriteJson(LatestPath, Pointer)
    return Pointer


def LoadEpochResume(
    ArtifactsRoot: Path,
    Seed: int,
    Model: LogMelCnn,
    Optimizer: torch.optim.Optimizer,
    Config: dict[str, Any],
    RunIdentity: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    _, LatestPath = ResumePaths(ArtifactsRoot, Seed)
    if not LatestPath.is_file():
        return 0, [], {
            "guarded_batch_count": 0,
            "maximum_gradient_norm": 0.0,
            "optimizer_skip_count": 0,
            "precision_mode": "fp32_guarded",
            "autocast_enabled": False,
            "grad_scaler_enabled": False,
            "parent_intraop_threads": 1,
            "parent_interop_threads": 1,
            "worker_intraop_threads": 1,
        }
    Pointer = StrictLoadJson(LatestPath)
    if Pointer.get("schema") != "e01-r5-resume-pointer-v1" or Pointer.get("seed") != Seed:
        raise RuntimeError(f"Invalid E01-R5 resume pointer: {LatestPath}")
    ValidateRunIdentity(RunIdentity, Pointer.get("run_identity"))
    CheckpointPath = (ArtifactsRoot / Pointer["checkpoint_relative_path"]).resolve()
    ArtifactsResolved = ArtifactsRoot.resolve()
    if ArtifactsResolved not in CheckpointPath.parents:
        raise RuntimeError("Resume checkpoint escaped the R5 artifacts directory")
    if not CheckpointPath.is_file() or HashFile(CheckpointPath) != Pointer["checkpoint_sha256"]:
        raise RuntimeError("Resume checkpoint missing or SHA mismatch")
    Payload = torch.load(CheckpointPath, map_location="cpu", weights_only=False)
    if Payload.get("schema") != ResumeSchemaVersion or Payload.get("seed") != Seed:
        raise RuntimeError("Resume checkpoint schema/seed mismatch")
    ValidateRunIdentity(RunIdentity, Payload.get("run_identity"))
    if Payload.get("config") != Config:
        raise RuntimeError("Resume checkpoint runtime config mismatch")
    NextEpoch = int(Payload["next_epoch"])
    if not 0 <= NextEpoch <= int(Config["epochs"]):
        raise RuntimeError("Resume checkpoint next_epoch is outside the fixed workload")
    EpochRows = Payload["epoch_rows"]
    NumericalSummary = Payload["numerical_summary"]
    AssertFinitePayload(EpochRows)
    AssertFinitePayload(NumericalSummary)
    if len(EpochRows) != NextEpoch or Pointer["epoch_rows_count"] != NextEpoch:
        raise RuntimeError("Resume checkpoint epoch-row count mismatch")
    RequireNestedTensorsFinite(Payload["model_state"], "$.model_state")
    RequireNestedTensorsFinite(Payload["optimizer_state"], "$.optimizer_state")
    Model.load_state_dict(Payload["model_state"], strict=True)
    Optimizer.load_state_dict(Payload["optimizer_state"])
    return NextEpoch, EpochRows, NumericalSummary


def CompleteSeedPath(ArtifactsRoot: Path, Seed: int) -> Path:
    return ArtifactsRoot / f"seed-{Seed}-complete.json"


def LoadCompleteSeedResult(
    DeepvoiceRoot: Path,
    ArtifactsRoot: Path,
    Seed: int,
    RunIdentity: dict[str, Any],
) -> dict[str, Any] | None:
    ResultPath = CompleteSeedPath(ArtifactsRoot, Seed)
    if not ResultPath.is_file():
        return None
    Payload = StrictLoadJson(ResultPath)
    AssertFinitePayload(Payload)
    if (
        Payload.get("schema") != CompleteSeedSchemaVersion
        or Payload.get("status") != "COMPLETE"
        or Payload.get("seed") != Seed
    ):
        raise RuntimeError(f"Invalid strict complete-seed result: {ResultPath}")
    ValidateRunIdentity(RunIdentity, Payload.get("run_identity"))
    ArtifactHashes = Payload.get("artifact_hashes")
    if not isinstance(ArtifactHashes, dict) or not ArtifactHashes:
        raise RuntimeError("Complete-seed result has no artifact hash inventory")
    DeepvoiceResolved = DeepvoiceRoot.resolve()
    for RelativePath, ExpectedSha in ArtifactHashes.items():
        ArtifactPath = (DeepvoiceRoot / RelativePath).resolve()
        if DeepvoiceResolved not in ArtifactPath.parents:
            raise RuntimeError("Complete-seed artifact escaped DeepVoice root")
        if not ArtifactPath.is_file() or HashFile(ArtifactPath) != ExpectedSha:
            raise RuntimeError(f"Complete-seed artifact SHA mismatch: {RelativePath}")
    return Payload


def WriteLiveProgress(
    ProgressPath: Path,
    RunIdentity: dict[str, Any],
    Payload: dict[str, Any],
) -> None:
    FullPayload = {
        "schema": "e01-r5-live-progress-v1",
        "status": "RUNNING",
        "updated_utc": UtcNow(),
        "run_identity": RunIdentity,
        **Payload,
    }
    AssertFinitePayload(FullPayload)
    AtomicWriteJson(ProgressPath, FullPayload)


def PrintTrainProgress(
    Seed: int,
    SeedIndex: int,
    SeedCount: int,
    Epoch: int,
    EpochCount: int,
    BatchIndex: int,
    BatchCount: int,
    CurrentLoss: float,
    MeanLoss: float,
    Progress: dict[str, float],
    Memory: dict[str, float],
) -> None:
    print(
        f"[TRAIN] seed={Seed} ({SeedIndex + 1}/{SeedCount}) "
        f"epoch={Epoch + 1}/{EpochCount} batch={BatchIndex + 1}/{BatchCount} "
        f"overall={Progress['overall_percent']:.2f}% "
        f"loss={CurrentLoss:.6f} mean_loss={MeanLoss:.6f} "
        f"samples/s={Progress['units_per_second']:.2f} "
        f"elapsed={FormatDuration(Progress['elapsed_seconds'])} "
        f"ETA={FormatDuration(Progress['eta_seconds'])} "
        f"CUDA_alloc={Memory['cuda_allocated_gib']:.3f}GiB "
        f"CUDA_reserved={Memory['cuda_reserved_gib']:.3f}GiB",
        flush=True,
    )


def PrintValidationProgress(
    Seed: int,
    SeedIndex: int,
    SeedCount: int,
    FileIndex: int,
    FileCount: int,
    SegmentCount: int,
    Progress: dict[str, float],
    Memory: dict[str, float],
) -> None:
    print(
        f"[VALID] seed={Seed} ({SeedIndex + 1}/{SeedCount}) "
        f"file={FileIndex + 1}/{FileCount} segments_done={SegmentCount} "
        f"overall={Progress['overall_percent']:.2f}% "
        f"segments/s={Progress['units_per_second']:.2f} "
        f"elapsed={FormatDuration(Progress['elapsed_seconds'])} "
        f"ETA={FormatDuration(Progress['eta_seconds'])} "
        f"CUDA_alloc={Memory['cuda_allocated_gib']:.3f}GiB "
        f"CUDA_reserved={Memory['cuda_reserved_gib']:.3f}GiB",
        flush=True,
    )


def RunSeedResumable(
    Seed: int,
    SeedIndex: int,
    TrainingRecords: list[AudioRecord],
    ValidationRecords: list[AudioRecord],
    Config: dict[str, Any],
    DeepvoiceRoot: Path,
    ArtifactsRoot: Path,
    ProgressPath: Path,
    RunIdentity: dict[str, Any],
    CompletedUnitsBeforeSeed: int,
    TotalUnits: int,
    OverallStarted: float,
) -> dict[str, Any]:
    PrecisionEvidence = RequireFp32TrainingMode(Config)
    ThreadEvidence = ConfigureParentCpuThreads(Config)
    SetSeed(Seed)
    Device = torch.device("cuda")
    Model = LogMelCnn(Config).to(Device)
    RequireModelParametersFinite(Model, f"seed_{Seed}", -1, "initial")
    Optimizer = torch.optim.AdamW(
        Model.parameters(),
        lr=float(Config["learning_rate"]),
        weight_decay=float(Config["weight_decay"]),
    )
    NextEpoch, EpochRows, NumericalSummary = LoadEpochResume(
        ArtifactsRoot,
        Seed,
        Model,
        Optimizer,
        Config,
        RunIdentity,
    )
    Dataset = TrainingDataset(TrainingRecords, Seed, Config)
    SamplesPerEpoch = int(Config["samples_per_epoch"])
    BaseSampler = GroupFirstBalancedSampler(TrainingRecords, SamplesPerEpoch, Seed)
    Sampler = EpochTaggedSampler(BaseSampler)
    WorkerCount = int(Config["workers"])
    Loader = DataLoader(
        Dataset,
        batch_size=int(Config["batch_size"]),
        sampler=Sampler,
        num_workers=WorkerCount,
        pin_memory=True,
        persistent_workers=WorkerCount > 0,
        prefetch_factor=2 if WorkerCount > 0 else None,
        worker_init_fn=PinDataLoaderWorkerCpuThreads,
    )
    EpochCount = int(Config["epochs"])
    BatchCountExpected = math.ceil(SamplesPerEpoch / int(Config["batch_size"]))
    PrintEvery = int(Config["progress_print_every_batches"])
    RunStarted = time.perf_counter()
    SessionTrainingSamples = 0
    if NextEpoch:
        print(
            f"[RESUME] seed={Seed} next_epoch={NextEpoch + 1}/{EpochCount} "
            f"checkpoint_epochs={NextEpoch}",
            flush=True,
        )
    for Epoch in range(NextEpoch, EpochCount):
        Dataset.SetEpoch(Epoch)
        Sampler.SetEpoch(Epoch)
        Model.train()
        LossTotal = 0.0
        EpochSamples = 0
        EpochBatchCount = 0
        EpochStarted = time.perf_counter()
        for BatchIndex, (
            RecordIndices,
            Waveforms,
            ValidSampleCounts,
            Labels,
            Masks,
        ) in enumerate(Loader):
            del RecordIndices
            Stage = f"seed_{Seed}_epoch_{Epoch}"
            RequireFiniteTensor(Waveforms, Stage, BatchIndex, "waveforms_cpu")
            Waveforms = Waveforms.to(Device, non_blocking=True)
            ValidSampleCounts = ValidSampleCounts.to(Device, non_blocking=True)
            Labels = Labels.to(Device, non_blocking=True)
            Masks = Masks.to(Device, non_blocking=True)
            RequireFiniteTensor(Waveforms, Stage, BatchIndex, "waveforms_gpu")
            RequireMaskedLabels(Labels, Masks, Stage, BatchIndex)
            Optimizer.zero_grad(set_to_none=True)
            Logits = Model(Waveforms, ValidSampleCounts)
            Loss = CalculateMaskedBalancedLoss(Logits, Labels, Masks)
            Evidence = GuardedFp32OptimizationStep(
                Model,
                Optimizer,
                Logits,
                Loss,
                Stage,
                BatchIndex,
            )
            CurrentLoss = float(Loss.detach().cpu())
            BatchSamples = int(Waveforms.shape[0])
            LossTotal += CurrentLoss
            EpochSamples += BatchSamples
            SessionTrainingSamples += BatchSamples
            EpochBatchCount += 1
            NumericalSummary["guarded_batch_count"] += 1
            NumericalSummary["maximum_gradient_norm"] = max(
                NumericalSummary["maximum_gradient_norm"], Evidence["gradient_norm"]
            )
            IsLogBatch = (BatchIndex + 1) % PrintEvery == 0 or (
                BatchIndex + 1 == BatchCountExpected
            )
            if IsLogBatch:
                CompletedUnits = (
                    CompletedUnitsBeforeSeed + Epoch * SamplesPerEpoch + EpochSamples
                )
                Progress = ProgressMath(
                    CompletedUnits,
                    TotalUnits,
                    SessionTrainingSamples,
                    max(time.perf_counter() - RunStarted, 1e-9),
                    time.perf_counter() - OverallStarted,
                )
                Memory = CudaMemory()
                MeanLoss = LossTotal / EpochBatchCount
                PrintTrainProgress(
                    Seed,
                    SeedIndex,
                    len(Config["seeds"]),
                    Epoch,
                    EpochCount,
                    BatchIndex,
                    BatchCountExpected,
                    CurrentLoss,
                    MeanLoss,
                    Progress,
                    Memory,
                )
                WriteLiveProgress(
                    ProgressPath,
                    RunIdentity,
                    {
                        "stage": "TRAIN",
                        "seed": Seed,
                        "seed_index": SeedIndex,
                        "epoch": Epoch + 1,
                        "batch": BatchIndex + 1,
                        "overall_percent": Progress["overall_percent"],
                        "current_loss": CurrentLoss,
                        "mean_loss": MeanLoss,
                        "samples_per_second": Progress["units_per_second"],
                        "elapsed_seconds": Progress["elapsed_seconds"],
                        "eta_seconds": Progress["eta_seconds"],
                        **Memory,
                    },
                )
        if EpochSamples != SamplesPerEpoch or EpochBatchCount != BatchCountExpected:
            raise RuntimeError("R5 epoch did not execute the fixed statistical workload")
        EpochRow = {
            "epoch": Epoch + 1,
            "mean_train_loss": LossTotal / EpochBatchCount,
            "samples": EpochSamples,
            "batches": EpochBatchCount,
            "seconds": time.perf_counter() - EpochStarted,
        }
        EpochRows.append(EpochRow)
        Pointer = SaveEpochResume(
            ArtifactsRoot,
            Seed,
            Epoch + 1,
            EpochRows,
            NumericalSummary,
            Model,
            Optimizer,
            Config,
            RunIdentity,
        )
        print(
            f"[CHECKPOINT] seed={Seed} epoch={Epoch + 1}/{EpochCount} "
            f"sha256={Pointer['checkpoint_sha256'][:16]}...",
            flush=True,
        )

    Model.eval()
    Predictions = []
    ValidationStarted = time.perf_counter()
    ValidationSegmentsSeen = 0
    ValidationEvery = int(Config["validation_progress_every_files"])
    for RecordIndex, Record in enumerate(ValidationRecords):
        Prediction = PredictRecord(Model, Record, Device, Config)
        if not np.isfinite(Prediction).all():
            raise RuntimeError(f"Seed {Seed} validation prediction is nonfinite")
        Predictions.append(Prediction)
        ValidationSegmentsSeen += ValidationSegmentCount(Record, Config)
        IsLogFile = (RecordIndex + 1) % ValidationEvery == 0 or (
            RecordIndex + 1 == len(ValidationRecords)
        )
        if IsLogFile:
            CompletedUnits = (
                CompletedUnitsBeforeSeed
                + EpochCount * SamplesPerEpoch
                + ValidationSegmentsSeen
            )
            Progress = ProgressMath(
                CompletedUnits,
                TotalUnits,
                (EpochCount - NextEpoch) * SamplesPerEpoch
                + ValidationSegmentsSeen,
                max(time.perf_counter() - RunStarted, 1e-9),
                time.perf_counter() - OverallStarted,
            )
            Memory = CudaMemory()
            PrintValidationProgress(
                Seed,
                SeedIndex,
                len(Config["seeds"]),
                RecordIndex,
                len(ValidationRecords),
                ValidationSegmentsSeen,
                Progress,
                Memory,
            )
            WriteLiveProgress(
                ProgressPath,
                RunIdentity,
                {
                    "stage": "VALID",
                    "seed": Seed,
                    "seed_index": SeedIndex,
                    "validation_file": RecordIndex + 1,
                    "validation_file_count": len(ValidationRecords),
                    "validation_segments_seen": ValidationSegmentsSeen,
                    "overall_percent": Progress["overall_percent"],
                    "segments_per_second": Progress["units_per_second"],
                    "elapsed_seconds": Progress["elapsed_seconds"],
                    "eta_seconds": Progress["eta_seconds"],
                    **Memory,
                },
            )
    PredictionArray = np.stack(Predictions)
    Labels = np.asarray([Record.Labels for Record in ValidationRecords], dtype=np.float64)
    Masks = np.asarray([Record.Masks for Record in ValidationRecords], dtype=bool)
    Metrics = CalculateHeadMetrics(Labels, Masks, PredictionArray)
    Proxy = CalculateCompetitionProxy(Metrics)
    MetricRows = [Record.ToMetricRow() for Record in ValidationRecords]
    ShortcutRows = BuildShortcutMetricAudit(
        MetricRows,
        Labels,
        Masks,
        PredictionArray,
        Scope=f"validation_seed_{Seed}",
    )
    BootstrapRows, BootstrapSummary = BootstrapByContentGroup(
        MetricRows,
        Labels,
        Masks,
        PredictionArray,
        Seed,
        int(Config["bootstrap_replicates"]),
        0.95,
    )
    Singleton = CheckModelSingleton(Model, Device, Config, Seed)
    if Singleton["status"] != "PASS":
        raise RuntimeError("R5 model singleton equivalence failed")
    ResultCore = {
        "epochs": EpochRows,
        "head_metrics": Metrics,
        "proxy": Proxy,
        "bootstrap": BootstrapSummary,
        "singleton": Singleton,
        "numerical_integrity": {
            **NumericalSummary,
            **PrecisionEvidence,
            **ThreadEvidence,
        },
        "validation_seconds": time.perf_counter() - ValidationStarted,
        "validation_segments": ValidationSegmentsSeen,
        "seed_runtime_seconds_this_launch": time.perf_counter() - RunStarted,
    }
    AssertFinitePayload(ResultCore)
    CheckpointPath = (
        DeepvoiceRoot
        / Config["final_checkpoints_relative_path"]
        / f"seed-{Seed}.pt"
    )
    FinalCheckpointPayload = {
        "schema": "e01-r5-final-model-v1",
        "experiment_id": "E01",
        "revision": "R5",
        "seed": Seed,
        "config": Config,
        "run_identity": RunIdentity,
        "model_state": Model.state_dict(),
        "metrics": Metrics,
        "proxy": Proxy,
    }
    RequireNestedTensorsFinite(FinalCheckpointPayload["model_state"], "$.model_state")
    AtomicTorchSave(CheckpointPath, FinalCheckpointPayload)
    OofPath = ArtifactsRoot / f"validation-oof-seed-{Seed}.csv.gz"
    ShortcutPath = ArtifactsRoot / f"shortcut-seed-{Seed}.csv.gz"
    BootstrapPath = ArtifactsRoot / f"bootstrap-seed-{Seed}.csv.gz"
    AtomicWriteGzipCsv(
        OofPath,
        BuildPredictionRows(ValidationRecords, PredictionArray, Seed),
    )
    AtomicWriteGzipCsv(ShortcutPath, ShortcutRows)
    AtomicWriteGzipCsv(BootstrapPath, BootstrapRows)
    ArtifactPaths = (CheckpointPath, OofPath, ShortcutPath, BootstrapPath)
    ArtifactHashes = {
        str(PathValue.relative_to(DeepvoiceRoot)).replace("\\", "/"): HashFile(
            PathValue
        )
        for PathValue in ArtifactPaths
    }
    CompletePayload = {
        "schema": CompleteSeedSchemaVersion,
        "status": "COMPLETE",
        "experiment_id": "E01",
        "revision": "R5",
        "seed": Seed,
        "run_identity": RunIdentity,
        "result": ResultCore,
        "artifact_hashes": ArtifactHashes,
        "completed_utc": UtcNow(),
    }
    AssertFinitePayload(CompletePayload)
    CompletePath = CompleteSeedPath(ArtifactsRoot, Seed)
    if CompletePath.exists():
        raise FileExistsError(f"Refusing to overwrite complete-seed result: {CompletePath}")
    AtomicWriteJson(CompletePath, CompletePayload)
    print(
        f"[SEED COMPLETE] seed={Seed} "
        f"RobustSelectionScore={Proxy['RobustSelectionScore']:.8f}",
        flush=True,
    )
    return CompletePayload


def BuildCompletionReport(Result: dict[str, Any]) -> str:
    Lines = [
        "EXPERIMENT_BATCH: COMPLETE",
        "",
        "# E01-R5 full three-seed training report",
        "",
        f"- manifest SHA-256: `{Result['run_identity']['manifest_sha256']}`",
        f"- code inventory SHA-256: `{Result['run_identity']['code_inventory_sha256']}`",
        "- precision: guarded FP32; autocast/GradScaler disabled",
        "- workload: 32,768 samples/epoch × 20 epochs × 3 seeds",
        "- workers/batch: 2 / 32",
        f"- RobustSelectionScore mean: {Result['score_mean']:.8f}",
        f"- RobustSelectionScore population std: {Result['score_std_population']:.8f}",
        f"- runtime seconds: {Result['runtime_seconds']:.3f}",
        "- test statistics/predictions/metrics: 0/0/0",
        "",
    ]
    for SeedResult in Result["seed_results"]:
        Lines.append(
            f"- seed {SeedResult['seed']}: "
            f"{SeedResult['result']['proxy']['RobustSelectionScore']:.8f}"
        )
    return "\n".join(Lines) + "\n"


def RunFullTrainingResumable(
    DeepvoiceRoot: Path,
    SourceRoot: Path,
    ConfigPath: Path,
    Config: dict[str, Any],
) -> dict[str, Any]:
    Started = time.perf_counter()
    ArtifactsRoot = DeepvoiceRoot / Config["resume_artifacts_relative_path"]
    ProgressPath = ArtifactsRoot / "progress.json"
    StatusPath = ArtifactsRoot / "status.json"
    RunIdentity: dict[str, Any] | None = None
    try:
        ValidateFixedR5Contract(Config)
        RequireFp32TrainingMode(Config)
        ThreadEvidence = ConfigureParentCpuThreads(Config)
        RunIdentity, CodeInventory = BuildRunIdentity(
            DeepvoiceRoot,
            SourceRoot,
            ConfigPath,
            Config,
        )
        AtomicWriteJson(
            StatusPath,
            {
                "schema": "e01-r5-status-v1",
                "status": "RUNNING",
                "updated_utc": UtcNow(),
                "run_identity": RunIdentity,
                "cpu_threads": ThreadEvidence,
            },
        )
        Preflight = RunPreflight(DeepvoiceRoot, Config)
        if Preflight["status"] != "READY":
            raise RuntimeError(f"E01-R5 preflight blocked: {Preflight['blockers']}")
        TrainingRecords, ValidationRecords, ManifestSummary = LoadE01Records(
            DeepvoiceRoot / Config["manifest_relative_path"]
        )
        NonTestRecords = [*TrainingRecords, *ValidationRecords]
        CacheGate = VerifyCompleteCache(
            NonTestRecords,
            DeepvoiceRoot / Config["cache_relative_path"],
        )
        if CacheGate["status"] != "PASS":
            raise RuntimeError(f"E01-R5 exact cache gate failed: {CacheGate}")
        SamplesPerSeed = int(Config["epochs"]) * int(Config["samples_per_epoch"])
        ValidationSegmentsPerSeed = sum(
            ValidationSegmentCount(Record, Config) for Record in ValidationRecords
        )
        UnitsPerSeed = SamplesPerSeed + ValidationSegmentsPerSeed
        TotalUnits = len(Config["seeds"]) * UnitsPerSeed
        SeedResults = []
        CompletedUnits = 0
        for SeedIndex, Seed in enumerate(Config["seeds"]):
            CompleteResult = LoadCompleteSeedResult(
                DeepvoiceRoot,
                ArtifactsRoot,
                int(Seed),
                RunIdentity,
            )
            if CompleteResult is not None:
                print(
                    f"[SKIP COMPLETE] seed={Seed} strict result and artifact SHA PASS",
                    flush=True,
                )
            else:
                CompleteResult = RunSeedResumable(
                    int(Seed),
                    SeedIndex,
                    TrainingRecords,
                    ValidationRecords,
                    Config,
                    DeepvoiceRoot,
                    ArtifactsRoot,
                    ProgressPath,
                    RunIdentity,
                    CompletedUnits,
                    TotalUnits,
                    Started,
                )
            SeedResults.append(CompleteResult)
            CompletedUnits += UnitsPerSeed
        Scores = np.asarray(
            [
                SeedResult["result"]["proxy"]["RobustSelectionScore"]
                for SeedResult in SeedResults
            ],
            dtype=np.float64,
        )
        Result = {
            "schema": "e01-r5-full-result-v1",
            "experiment_batch": "COMPLETE",
            "experiment_id": "E01",
            "revision": "R5",
            "run_identity": RunIdentity,
            "manifest_integrity": ManifestSummary,
            "test_usage_contract": {
                "allowed_fields": [
                    "content_group_key",
                    "recommended_content_split",
                ],
                "test_statistics": 0,
                "test_predictions": 0,
                "test_metrics": 0,
            },
            "fixed_workload": {
                "samples_per_epoch": int(Config["samples_per_epoch"]),
                "epochs": int(Config["epochs"]),
                "seeds": list(Config["seeds"]),
                "batch_size": int(Config["batch_size"]),
                "workers": int(Config["workers"]),
            },
            "seed_results": SeedResults,
            "score_mean": float(np.mean(Scores)),
            "score_std_population": float(np.std(Scores, ddof=0)),
            "success_std_le_0_005": bool(np.std(Scores, ddof=0) <= 0.005),
            "runtime_seconds": time.perf_counter() - Started,
            "completed_utc": UtcNow(),
        }
        AssertFinitePayload(Result)
        ResultPath = DeepvoiceRoot / Config["full_result_relative_path"]
        ReportPath = DeepvoiceRoot / Config["full_report_relative_path"]
        InventoryPath = DeepvoiceRoot / "reports/e01-r5-code-inventory.csv"
        if ResultPath.exists():
            Existing = StrictLoadJson(ResultPath)
            AssertFinitePayload(Existing)
            ValidateRunIdentity(RunIdentity, Existing.get("run_identity"))
            if Existing.get("experiment_batch") != "COMPLETE":
                raise RuntimeError("Existing R5 top-level result is not COMPLETE")
            Result = Existing
            if not ReportPath.exists():
                AtomicWriteBytes(
                    ReportPath,
                    BuildCompletionReport(Result).encode("utf-8"),
                )
            if not InventoryPath.exists():
                AtomicWriteBytes(InventoryPath, CsvBytes(CodeInventory))
        else:
            if ReportPath.exists() or InventoryPath.exists():
                raise FileExistsError("Partial R5 top-level completion outputs exist")
            AtomicWriteJson(ResultPath, Result)
            AtomicWriteBytes(ReportPath, BuildCompletionReport(Result).encode("utf-8"))
            AtomicWriteBytes(InventoryPath, CsvBytes(CodeInventory))
        AtomicWriteJson(
            ProgressPath,
            {
                "schema": "e01-r5-live-progress-v1",
                "status": "COMPLETE",
                "updated_utc": UtcNow(),
                "run_identity": RunIdentity,
                "overall_percent": 100.0,
                "elapsed_seconds": time.perf_counter() - Started,
                "eta_seconds": 0.0,
            },
        )
        AtomicWriteJson(
            StatusPath,
            {
                "schema": "e01-r5-status-v1",
                "status": "COMPLETE",
                "updated_utc": UtcNow(),
                "run_identity": RunIdentity,
                "result_relative_path": Config["full_result_relative_path"],
            },
        )
        print(
            f"[COMPLETE] result={ResultPath} report={ReportPath}",
            flush=True,
        )
        return Result
    except KeyboardInterrupt:
        AtomicWriteJson(
            StatusPath,
            {
                "schema": "e01-r5-status-v1",
                "status": "INTERRUPTED",
                "updated_utc": UtcNow(),
                "run_identity": RunIdentity,
                "elapsed_seconds": time.perf_counter() - Started,
                "resume_instruction": "Relaunch the same CLI; latest completed epoch is retained.",
            },
        )
        print(
            "[INTERRUPTED] latest completed epoch checkpoint retained; relaunch the same command.",
            flush=True,
        )
        raise
    except Exception as Error:
        AtomicWriteJson(
            StatusPath,
            {
                "schema": "e01-r5-status-v1",
                "status": "FAILED",
                "updated_utc": UtcNow(),
                "run_identity": RunIdentity,
                "elapsed_seconds": time.perf_counter() - Started,
                "error_type": type(Error).__name__,
                "error": str(Error),
            },
        )
        print(f"[FAILED] {type(Error).__name__}: {Error}", flush=True)
        raise
