# /// <summary>
# Full three-seed E01 training implementation guarded by the resource preflight
# /// </summary>

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .audio import (
    CreateTrainingSegmentWithLength,
    CreateValidationSegmentsWithLengths,
    LoadLocatorWaveform,
)
from .contract_adapter import (
    BootstrapByContentGroup,
    BuildShortcutMetricAudit,
    CalculateCompetitionProxy,
    CalculateHeadMetrics,
    HeadNames,
    HeadWeights,
)
from .model import LogMelCnn
from .preflight import RunPreflight
from .records import AudioRecord, LoadE01Records
from .sampling import GroupFirstBalancedSampler


def SetSeed(Seed: int) -> None:
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def StableOffset(Seed: int, Epoch: int, Record: AudioRecord) -> float:
    Payload = f"{Seed}|{Epoch}|{Record.Dataset}|{Record.SampleId}".encode("utf-8")
    Value = int.from_bytes(hashlib.sha256(Payload).digest()[:8], "big")
    return Value / float(2**64 - 1)


def AugmentWaveform(Waveform: torch.Tensor, Seed: int) -> torch.Tensor:
    Generator = torch.Generator().manual_seed(Seed)
    GainDb = float(torch.empty(1).uniform_(-8.0, 5.0, generator=Generator))
    Output = Waveform * (10.0 ** (GainDb / 20.0))
    if float(torch.rand((), generator=Generator)) < 0.5:
        SignalRootMeanSquare = Output.square().mean().sqrt().clamp_min(1e-5)
        SignalToNoiseDb = float(
            torch.empty(1).uniform_(15.0, 35.0, generator=Generator)
        )
        NoiseScale = SignalRootMeanSquare / (10.0 ** (SignalToNoiseDb / 20.0))
        Noise = torch.randn(Output.shape, generator=Generator, dtype=Output.dtype)
        Output = Output + Noise * NoiseScale
    return Output.clamp(-1.0, 1.0)


class TrainingDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        Records: Sequence[AudioRecord],
        Seed: int,
        Config: dict[str, Any],
    ) -> None:
        self.Records = list(Records)
        self.Seed = Seed
        self.Epoch = 0
        self.SampleRate = int(Config["sample_rate"])
        self.SegmentSeconds = float(Config["segment_seconds"])

    def SetEpoch(self, Epoch: int) -> None:
        self.Epoch = Epoch

    def __len__(self) -> int:
        return len(self.Records)

    def __getitem__(
        self,
        Index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Record = self.Records[Index]
        Waveform = LoadLocatorWaveform(Record.Locator, self.SampleRate)
        Offset = StableOffset(self.Seed, self.Epoch, Record)
        Segment, ValidSampleCount = CreateTrainingSegmentWithLength(
            Waveform,
            Offset,
            self.SampleRate,
            self.SegmentSeconds,
        )
        AugmentSeed = self.Seed + self.Epoch * 1_000_003 + Index
        Segment = AugmentWaveform(Segment, AugmentSeed)
        Labels = torch.tensor(Record.Labels, dtype=torch.float32)
        Masks = torch.tensor(Record.Masks, dtype=torch.bool)
        return Segment, torch.tensor(ValidSampleCount), Labels, Masks


def CalculateMaskedBalancedLoss(
    Logits: torch.Tensor,
    Labels: torch.Tensor,
    Masks: torch.Tensor,
) -> torch.Tensor:
    SafeLabels = torch.where(Masks, Labels, torch.zeros_like(Labels))
    ElementLoss = F.binary_cross_entropy_with_logits(
        Logits,
        SafeLabels,
        reduction="none",
    )
    HeadLosses = []
    for HeadIndex in range(len(HeadNames)):
        Observed = Masks[:, HeadIndex]
        if not torch.any(Observed):
            HeadLosses.append(Logits[:, HeadIndex].sum() * 0.0)
            continue
        Targets = SafeLabels[Observed, HeadIndex]
        Losses = ElementLoss[Observed, HeadIndex]
        Positive = Targets == 1.0
        Negative = Targets == 0.0
        ClassLosses = []
        if torch.any(Positive):
            ClassLosses.append(Losses[Positive].mean())
        if torch.any(Negative):
            ClassLosses.append(Losses[Negative].mean())
        HeadLosses.append(torch.stack(ClassLosses).mean())
    Weights = torch.as_tensor(HeadWeights, device=Logits.device, dtype=Logits.dtype)
    return (torch.stack(HeadLosses) * Weights).sum()


@torch.inference_mode()
def PredictRecord(
    Model: LogMelCnn,
    Record: AudioRecord,
    Device: torch.device,
    Config: dict[str, Any],
) -> np.ndarray:
    Waveform = LoadLocatorWaveform(Record.Locator, int(Config["sample_rate"]))
    Segments, ValidSampleCounts = CreateValidationSegmentsWithLengths(
        Waveform,
        int(Config["sample_rate"]),
        float(Config["segment_seconds"]),
        int(Config["max_segments_per_file"]),
    )
    SegmentPredictions = []
    BatchSize = int(Config["batch_size"])
    for Start in range(0, Segments.shape[0], BatchSize):
        Logits = Model(
            Segments[Start : Start + BatchSize].to(Device),
            ValidSampleCounts[Start : Start + BatchSize].to(Device),
        )
        SegmentPredictions.append(torch.sigmoid(Logits).cpu())
    return torch.cat(SegmentPredictions).mean(dim=0).numpy().astype(np.float64)


@torch.inference_mode()
def CheckModelSingleton(
    Model: LogMelCnn,
    Device: torch.device,
    Config: dict[str, Any],
    Seed: int,
) -> dict[str, Any]:
    Generator = torch.Generator().manual_seed(Seed)
    SegmentCounts = (1, 3, 8, 2)
    FileSegments = [
        torch.randn(
            Count,
            round(int(Config["sample_rate"]) * float(Config["segment_seconds"])),
            generator=Generator,
        )
        * 0.05
        for Count in SegmentCounts
    ]
    Singleton = []
    for Segments in FileSegments:
        ValidCounts = torch.full(
            (Segments.shape[0],),
            Segments.shape[1],
            dtype=torch.long,
            device=Device,
        )
        Singleton.append(
            torch.sigmoid(Model(Segments.to(Device), ValidCounts)).mean(dim=0).cpu()
        )
    FlatSegments = torch.cat(FileSegments).to(Device)
    FlatValidCounts = torch.full(
        (FlatSegments.shape[0],),
        FlatSegments.shape[1],
        dtype=torch.long,
        device=Device,
    )
    FlatPredictions = torch.sigmoid(Model(FlatSegments, FlatValidCounts)).cpu()
    Batched = []
    Start = 0
    for Count in SegmentCounts:
        Batched.append(FlatPredictions[Start : Start + Count].mean(dim=0))
        Start += Count
    Delta = float(torch.max(torch.abs(torch.stack(Singleton) - torch.stack(Batched))))
    return {
        "seed": Seed,
        "file_count": len(SegmentCounts),
        "max_absolute_delta": Delta,
        "tolerance": float(Config["singleton_tolerance"]),
        "status": "PASS" if Delta <= float(Config["singleton_tolerance"]) else "FAIL",
    }


def WriteGzipCsv(OutputPath: Path, Rows: Sequence[dict[str, Any]]) -> None:
    if not Rows:
        raise ValueError("Cannot write empty prediction artifact")
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    Fields = list(Rows[0])
    with OutputPath.open("wb") as RawFile:
        with gzip.GzipFile(fileobj=RawFile, mode="wb", mtime=0) as GzipFile:
            with io.TextIOWrapper(GzipFile, encoding="utf-8", newline="") as TextFile:
                Writer = csv.DictWriter(TextFile, fieldnames=Fields)
                Writer.writeheader()
                Writer.writerows(Rows)


def BuildPredictionRows(
    Records: Sequence[AudioRecord],
    Predictions: np.ndarray,
    Seed: int,
) -> list[dict[str, Any]]:
    Rows = []
    for RecordIndex, Record in enumerate(Records):
        Row: dict[str, Any] = {
            "seed": Seed,
            "prediction_role": "fixed_validation_out_of_training_split",
            **Record.ToMetricRow(),
        }
        for HeadIndex, HeadName in enumerate(HeadNames):
            Row[f"{HeadName}_label"] = (
                Record.Labels[HeadIndex] if Record.Masks[HeadIndex] else ""
            )
            Row[f"{HeadName}_mask"] = Record.Masks[HeadIndex]
            Row[f"{HeadName}_prediction"] = float(Predictions[RecordIndex, HeadIndex])
        Rows.append(Row)
    return Rows


def RunSeed(
    Seed: int,
    TrainingRecords: list[AudioRecord],
    ValidationRecords: list[AudioRecord],
    Config: dict[str, Any],
    DeepvoiceRoot: Path,
) -> dict[str, Any]:
    SetSeed(Seed)
    Device = torch.device("cuda")
    Model = LogMelCnn(Config).to(Device)
    Dataset = TrainingDataset(TrainingRecords, Seed, Config)
    SamplesPerEpoch = int(Config["samples_per_epoch"])
    ExpectedSamplesPerEpoch = (
        int(Config["balanced_group_draws_per_epoch"])
        * int(Config["samples_per_balanced_group_draw"])
    )
    if SamplesPerEpoch != ExpectedSamplesPerEpoch:
        raise RuntimeError("E01 samples-per-epoch contract is internally inconsistent")
    Sampler = GroupFirstBalancedSampler(TrainingRecords, SamplesPerEpoch, Seed)
    Loader = DataLoader(
        Dataset,
        batch_size=int(Config["batch_size"]),
        sampler=Sampler,
        num_workers=int(Config["workers"]),
        pin_memory=True,
    )
    Optimizer = torch.optim.AdamW(
        Model.parameters(),
        lr=float(Config["learning_rate"]),
        weight_decay=float(Config["weight_decay"]),
    )
    Scaler = torch.amp.GradScaler("cuda", enabled=True)
    EpochRows = []
    SeedStarted = time.perf_counter()
    for Epoch in range(int(Config["epochs"])):
        Dataset.SetEpoch(Epoch)
        Sampler.SetEpoch(Epoch)
        Model.train()
        LossTotal = 0.0
        BatchCount = 0
        EpochStarted = time.perf_counter()
        for Waveforms, ValidSampleCounts, Labels, Masks in Loader:
            Waveforms = Waveforms.to(Device, non_blocking=True)
            ValidSampleCounts = ValidSampleCounts.to(Device, non_blocking=True)
            Labels = Labels.to(Device, non_blocking=True)
            Masks = Masks.to(Device, non_blocking=True)
            Optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                Logits = Model(Waveforms, ValidSampleCounts)
                Loss = CalculateMaskedBalancedLoss(Logits, Labels, Masks)
            Scaler.scale(Loss).backward()
            Scaler.unscale_(Optimizer)
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 5.0)
            Scaler.step(Optimizer)
            Scaler.update()
            LossTotal += float(Loss.detach().cpu())
            BatchCount += 1
        EpochRows.append(
            {
                "epoch": Epoch + 1,
                "mean_train_loss": LossTotal / max(BatchCount, 1),
                "seconds": time.perf_counter() - EpochStarted,
            }
        )

    Model.eval()
    ValidationStarted = time.perf_counter()
    Predictions = np.stack(
        [PredictRecord(Model, Record, Device, Config) for Record in ValidationRecords]
    )
    ValidationSeconds = time.perf_counter() - ValidationStarted
    Labels = np.asarray([Record.Labels for Record in ValidationRecords], dtype=np.float64)
    Masks = np.asarray([Record.Masks for Record in ValidationRecords], dtype=bool)
    Metrics = CalculateHeadMetrics(Labels, Masks, Predictions)
    Proxy = CalculateCompetitionProxy(Metrics)
    MetricRows = [Record.ToMetricRow() for Record in ValidationRecords]
    ShortcutRows = BuildShortcutMetricAudit(
        MetricRows,
        Labels,
        Masks,
        Predictions,
        Scope=f"validation_seed_{Seed}",
    )
    BootstrapRows, BootstrapSummary = BootstrapByContentGroup(
        MetricRows,
        Labels,
        Masks,
        Predictions,
        Seed,
        int(Config["bootstrap_replicates"]),
        0.95,
    )
    Singleton = CheckModelSingleton(Model, Device, Config, Seed)
    if Singleton["status"] != "PASS":
        raise RuntimeError("Model singleton equivalence failed")

    CheckpointPath = DeepvoiceRoot / "checkpoints" / "e01" / f"seed-{Seed}.pt"
    CheckpointPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": "E01",
            "seed": Seed,
            "config": Config,
            "model_state": Model.state_dict(),
            "metrics": Metrics,
            "proxy": Proxy,
        },
        CheckpointPath,
    )
    ArtifactsRoot = DeepvoiceRoot / "artifacts" / "e01"
    WriteGzipCsv(
        ArtifactsRoot / f"validation-oof-seed-{Seed}.csv.gz",
        BuildPredictionRows(ValidationRecords, Predictions, Seed),
    )
    WriteGzipCsv(ArtifactsRoot / f"shortcut-seed-{Seed}.csv.gz", ShortcutRows)
    WriteGzipCsv(ArtifactsRoot / f"bootstrap-seed-{Seed}.csv.gz", BootstrapRows)
    return {
        "seed": Seed,
        "epochs": EpochRows,
        "head_metrics": Metrics,
        "proxy": Proxy,
        "bootstrap": BootstrapSummary,
        "singleton": Singleton,
        "validation_seconds": ValidationSeconds,
        "validation_seconds_per_file": ValidationSeconds / len(ValidationRecords),
        "total_seconds": time.perf_counter() - SeedStarted,
        "checkpoint": str(CheckpointPath.relative_to(DeepvoiceRoot)),
    }


def RunFullTraining(DeepvoiceRoot: Path, Config: dict[str, Any]) -> dict[str, Any]:
    Preflight = RunPreflight(DeepvoiceRoot, Config)
    if Preflight["status"] != "READY":
        raise RuntimeError(
            "Full E01 training is blocked by preflight: "
            + ", ".join(Preflight["blockers"])
        )
    TrainingRecords, ValidationRecords, ManifestSummary = LoadE01Records(
        DeepvoiceRoot / Config["manifest_relative_path"]
    )
    SeedRuns = [
        RunSeed(Seed, TrainingRecords, ValidationRecords, Config, DeepvoiceRoot)
        for Seed in Config["seeds"]
    ]
    Scores = np.asarray(
        [Run["proxy"]["RobustSelectionScore"] for Run in SeedRuns],
        dtype=np.float64,
    )
    return {
        "experiment_batch": "COMPLETE",
        "experiment_id": "E01",
        "manifest": ManifestSummary,
        "seed_runs": SeedRuns,
        "score_mean": float(np.mean(Scores)),
        "score_std_population": float(np.std(Scores, ddof=0)),
        "success_std_le_0_005": bool(np.std(Scores, ddof=0) <= 0.005),
    }
