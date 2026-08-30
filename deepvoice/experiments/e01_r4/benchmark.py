# /// <summary>
# Windows worker, cached end-to-end GPU and conservative R4 runtime benchmarks
# /// </summary>

from __future__ import annotations

import gc
import hashlib
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .determinism import ConfigureParentCpuThreads, PinDataLoaderWorkerCpuThreads
from .model import LogMelCnn
from .numerical import (
    GuardedFp32OptimizationStep,
    RequireFiniteTensor,
    RequireFp32TrainingMode,
    RequireMaskedLabels,
)
from .records import AudioRecord
from .sampling import GroupFirstBalancedSampler
from .train_e01 import CalculateMaskedBalancedLoss, EpochTaggedSampler, TrainingDataset


def BenchmarkGpuBatch(Config: dict[str, Any]) -> dict[str, Any]:
    PrecisionEvidence = RequireFp32TrainingMode(Config)
    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "CUDA unavailable", "candidates": []}
    Device = torch.device("cuda")
    CandidateRows = []
    SegmentLength = round(int(Config["sample_rate"]) * float(Config["segment_seconds"]))
    TotalMemory = torch.cuda.get_device_properties(0).total_memory
    for BatchSize in (4, 8, 12, 16, 24, 32):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.manual_seed(20260830)
            Model = LogMelCnn(Config).to(Device).train()
            Optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-4)
            Waveforms = torch.randn(
                BatchSize, SegmentLength, device=Device, dtype=torch.float32
            ) * 0.01
            ValidSampleCounts = torch.full(
                (BatchSize,), SegmentLength, device=Device, dtype=torch.long
            )
            for Iteration in range(4):
                Optimizer.zero_grad(set_to_none=True)
                Logits = Model(Waveforms, ValidSampleCounts)
                Loss = Logits.square().mean()
                Evidence = GuardedFp32OptimizationStep(
                    Model,
                    Optimizer,
                    Logits,
                    Loss,
                    f"gpu_autotune_batch_{BatchSize}",
                    Iteration,
                )
                if Iteration == 0:
                    torch.cuda.synchronize()
                    Started = time.perf_counter()
            torch.cuda.synchronize()
            Seconds = time.perf_counter() - Started
            Iterations = 3
            PeakBytes = torch.cuda.max_memory_allocated()
            CandidateRows.append(
                {
                    "batch_size": BatchSize,
                    "status": "PASS",
                    "iterations": Iterations,
                    "seconds": Seconds,
                    "segments_per_second": BatchSize * Iterations / Seconds,
                    "peak_allocated_bytes": PeakBytes,
                    "peak_allocated_gib": PeakBytes / 1024**3,
                    "memory_fraction": PeakBytes / TotalMemory,
                    "final_gradient_norm": Evidence["gradient_norm"],
                    **PrecisionEvidence,
                }
            )
            del Model, Optimizer, Waveforms, ValidSampleCounts, Loss
        except torch.cuda.OutOfMemoryError as Error:
            CandidateRows.append(
                {"batch_size": BatchSize, "status": "OOM", "error": str(Error)}
            )
            torch.cuda.empty_cache()
            break
    EligibleRows = [
        Row
        for Row in CandidateRows
        if Row["status"] == "PASS" and Row["memory_fraction"] <= 0.85
    ]
    if not EligibleRows:
        return {
            "status": "BLOCKED",
            "reason": "No batch passed numerical and 85% VRAM gates",
            "candidates": CandidateRows,
        }
    RecommendedRow = max(
        EligibleRows,
        key=lambda Row: (Row["segments_per_second"], -Row["batch_size"]),
    )
    return {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "total_memory_bytes": TotalMemory,
        "selection_rule": "highest guarded segments/s among batches using <=85% VRAM",
        "recommended_batch_size": RecommendedRow["batch_size"],
        "recommended_segments_per_second": RecommendedRow["segments_per_second"],
        "recommended_peak_allocated_bytes": RecommendedRow["peak_allocated_bytes"],
        **PrecisionEvidence,
        "candidates": CandidateRows,
    }


def BuildTrainingLoader(
    Records: Sequence[AudioRecord],
    Config: dict[str, Any],
    Seed: int,
    SampleCount: int,
    WorkerCount: int,
) -> tuple[DataLoader, EpochTaggedSampler]:
    Dataset = TrainingDataset(Records, Seed, Config)
    BaseSampler = GroupFirstBalancedSampler(Records, SampleCount, Seed)
    Sampler = EpochTaggedSampler(BaseSampler)
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
    return Loader, Sampler


def ExpectedLocatorSequence(
    Records: Sequence[AudioRecord],
    SampleCount: int,
    Seed: int,
    Epoch: int = 0,
) -> tuple[list[int], str]:
    Sampler = GroupFirstBalancedSampler(Records, SampleCount, Seed)
    Sampler.SetEpoch(Epoch)
    Indices = list(iter(Sampler))
    Digest = hashlib.sha256()
    for Index in Indices:
        Digest.update(Records[Index].Locator.encode("utf-8"))
        Digest.update(b"\0")
    return Indices, Digest.hexdigest()


def ConsumeLoader(Loader: DataLoader, Records: Sequence[AudioRecord]) -> dict[str, Any]:
    Started = time.perf_counter()
    IndexSequence = []
    LocatorDigest = hashlib.sha256()
    TensorDigest = hashlib.sha256()
    SampleCount = 0
    for RecordIndices, Waveforms, ValidCounts, Labels, Masks in Loader:
        RequireFiniteTensor(Waveforms, "worker_benchmark", SampleCount, "waveforms")
        RequireMaskedLabels(Labels, Masks, "worker_benchmark", SampleCount)
        Indices = [int(Value) for Value in RecordIndices.tolist()]
        IndexSequence.extend(Indices)
        for Index in Indices:
            LocatorDigest.update(Records[Index].Locator.encode("utf-8"))
            LocatorDigest.update(b"\0")
        TensorDigest.update(Waveforms.contiguous().numpy().tobytes())
        TensorDigest.update(ValidCounts.contiguous().numpy().tobytes())
        CanonicalLabels = torch.where(Masks, Labels, torch.zeros_like(Labels))
        TensorDigest.update(CanonicalLabels.contiguous().numpy().tobytes())
        TensorDigest.update(Masks.contiguous().numpy().tobytes())
        SampleCount += Waveforms.shape[0]
    Seconds = time.perf_counter() - Started
    return {
        "sample_count": SampleCount,
        "seconds": Seconds,
        "samples_per_second": SampleCount / Seconds,
        "index_sequence": IndexSequence,
        "locator_sequence_sha256": LocatorDigest.hexdigest(),
        "tensor_sequence_sha256": TensorDigest.hexdigest(),
    }


def BenchmarkWindowsWorkers(
    Records: Sequence[AudioRecord], Config: dict[str, Any]
) -> dict[str, Any]:
    ThreadEvidence = ConfigureParentCpuThreads(Config)
    Seed = 20260830
    SampleCount = int(Config["worker_benchmark_samples"])
    ExpectedWarmIndices, ExpectedWarmLocatorDigest = ExpectedLocatorSequence(
        Records, SampleCount, Seed, 0
    )
    ExpectedMeasuredIndices, ExpectedMeasuredLocatorDigest = ExpectedLocatorSequence(
        Records, SampleCount, Seed, 1
    )
    Rows = []
    for WorkerCount in Config["worker_candidates"]:
        Loader, Sampler = BuildTrainingLoader(
            Records, Config, Seed, SampleCount, int(WorkerCount)
        )
        Sampler.SetEpoch(0)
        Warm = ConsumeLoader(Loader, Records)
        Sampler.SetEpoch(1)
        Measured = ConsumeLoader(Loader, Records)
        Stable = (
            Warm["index_sequence"] == ExpectedWarmIndices
            and Measured["index_sequence"] == ExpectedMeasuredIndices
            and Warm["locator_sequence_sha256"] == ExpectedWarmLocatorDigest
            and Measured["locator_sequence_sha256"] == ExpectedMeasuredLocatorDigest
        )
        Rows.append(
            {
                "workers": int(WorkerCount),
                "persistent_workers": int(WorkerCount) > 0,
                "status": "PASS" if Stable else "FAIL",
                "warm_seconds": Warm["seconds"],
                "warm_samples_per_second": Warm["samples_per_second"],
                "warm_tagged_epoch": 0,
                "warm_locator_sequence_sha256": Warm["locator_sequence_sha256"],
                "measured_seconds": Measured["seconds"],
                "measured_samples_per_second": Measured["samples_per_second"],
                "measured_tagged_epoch": 1,
                "locator_sequence_sha256": Measured["locator_sequence_sha256"],
                "tensor_sequence_sha256": Measured["tensor_sequence_sha256"],
                "exact_expected_locator_sequence": Stable,
                **ThreadEvidence,
            }
        )
        del Loader
        gc.collect()
    PassingRows = [Row for Row in Rows if Row["status"] == "PASS"]
    if not PassingRows:
        return {
            "status": "BLOCKED",
            "reason": "No worker candidate preserved exact deterministic sequence",
            "candidates": Rows,
        }
    ReferenceTensorDigest = PassingRows[0]["tensor_sequence_sha256"]
    if not all(
        Row["tensor_sequence_sha256"] == ReferenceTensorDigest for Row in PassingRows
    ):
        return {
            "status": "BLOCKED",
            "reason": "Worker candidates changed deterministic segment sequence",
            "candidates": Rows,
        }
    Recommended = max(
        PassingRows,
        key=lambda Row: (Row["measured_samples_per_second"], -Row["workers"]),
    )
    return {
        "status": "PASS",
        "sample_count_per_pass": SampleCount,
        "warm_tagged_epoch": 0,
        "measured_tagged_epoch": 1,
        "expected_warm_locator_sequence_sha256": ExpectedWarmLocatorDigest,
        "expected_measured_locator_sequence_sha256": ExpectedMeasuredLocatorDigest,
        "cross_worker_tensor_sequence_exact": True,
        "tensor_digest_fields": [
            "waveform_float32_bytes",
            "valid_sample_count",
            "canonical_observed_labels_masked_missing_to_zero",
            "label_masks",
        ],
        "canonical_masked_label_storage": True,
        **ThreadEvidence,
        "recommended_workers": Recommended["workers"],
        "recommended_samples_per_second": Recommended["measured_samples_per_second"],
        "candidates": Rows,
    }


def BenchmarkCachedEndToEndPilot(
    Records: Sequence[AudioRecord], Config: dict[str, Any]
) -> dict[str, Any]:
    PrecisionEvidence = RequireFp32TrainingMode(Config)
    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "CUDA unavailable"}
    Seed = 20260830
    SampleCount = int(Config["cached_gpu_pilot_samples"])
    WorkerCount = int(Config["workers"])
    Loader, Sampler = BuildTrainingLoader(
        Records, Config, Seed, SampleCount, WorkerCount
    )
    Sampler.SetEpoch(0)
    Device = torch.device("cuda")
    torch.manual_seed(Seed)
    Model = LogMelCnn(Config).to(Device).train()
    Optimizer = torch.optim.AdamW(
        Model.parameters(),
        lr=float(Config["learning_rate"]),
        weight_decay=float(Config["weight_decay"]),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    Started = time.perf_counter()
    LocatorDigest = hashlib.sha256()
    BatchCount = 0
    GradientNormMaximum = 0.0
    FinalLoss = None
    for RecordIndices, Waveforms, ValidCounts, Labels, Masks in Loader:
        for Index in RecordIndices.tolist():
            LocatorDigest.update(Records[int(Index)].Locator.encode("utf-8"))
            LocatorDigest.update(b"\0")
        RequireFiniteTensor(Waveforms, "cached_gpu_pilot", BatchCount, "waveforms_cpu")
        Waveforms = Waveforms.to(Device, non_blocking=True)
        ValidCounts = ValidCounts.to(Device, non_blocking=True)
        Labels = Labels.to(Device, non_blocking=True)
        Masks = Masks.to(Device, non_blocking=True)
        RequireMaskedLabels(Labels, Masks, "cached_gpu_pilot", BatchCount)
        Optimizer.zero_grad(set_to_none=True)
        Logits = Model(Waveforms, ValidCounts)
        Loss = CalculateMaskedBalancedLoss(Logits, Labels, Masks)
        Evidence = GuardedFp32OptimizationStep(
            Model,
            Optimizer,
            Logits,
            Loss,
            "cached_gpu_pilot",
            BatchCount,
        )
        GradientNormMaximum = max(GradientNormMaximum, Evidence["gradient_norm"])
        FinalLoss = float(Loss.detach().cpu())
        BatchCount += 1
    torch.cuda.synchronize()
    Seconds = time.perf_counter() - Started
    PeakBytes = torch.cuda.max_memory_allocated()
    _, ExpectedLocatorDigest = ExpectedLocatorSequence(Records, SampleCount, Seed)
    ObservedLocatorDigest = LocatorDigest.hexdigest()
    if ObservedLocatorDigest != ExpectedLocatorDigest:
        raise RuntimeError("Cached GPU pilot changed deterministic locator sequence")
    if FinalLoss is None:
        raise RuntimeError("Cached GPU pilot produced no batches")
    return {
        "status": "PASS",
        "scope": "guarded cached end-to-end data loading plus GPU optimization pilot",
        "is_e01_performance_result": False,
        "seed": Seed,
        "sample_count": SampleCount,
        "batch_count": BatchCount,
        "batch_size": int(Config["batch_size"]),
        "workers": WorkerCount,
        "persistent_workers": WorkerCount > 0,
        "seconds": Seconds,
        "end_to_end_samples_per_second": SampleCount / Seconds,
        "locator_sequence_sha256": ObservedLocatorDigest,
        "exact_expected_locator_sequence": True,
        "final_loss": FinalLoss,
        "maximum_fp32_gradient_norm": GradientNormMaximum,
        "optimizer_skip_count": 0,
        "guarded_batch_count": BatchCount,
        **PrecisionEvidence,
        "peak_allocated_bytes": PeakBytes,
        "peak_allocated_gib": PeakBytes / 1024**3,
    }


def ProjectRuntime(
    Config: dict[str, Any],
    Pilot: dict[str, Any],
    ValidationRecords: Sequence[AudioRecord],
    CacheGate: dict[str, Any],
) -> dict[str, Any]:
    SamplesPerEpoch = int(Config["samples_per_epoch"])
    ExpectedSamplesPerEpoch = (
        int(Config["balanced_group_draws_per_epoch"])
        * int(Config["samples_per_balanced_group_draw"])
    )
    if SamplesPerEpoch != ExpectedSamplesPerEpoch:
        raise RuntimeError("E01 samples-per-epoch contract is internally inconsistent")
    TrainingSamplesPerSeed = int(Config["epochs"]) * SamplesPerEpoch
    ValidationSegments = sum(
        min(
            int(Config["max_segments_per_file"]),
            max(1, int(np.ceil(Record.DurationSeconds / float(Config["segment_seconds"])))),
        )
        for Record in ValidationRecords
    )
    MeasuredRate = float(Pilot["end_to_end_samples_per_second"])
    SafetyFactor = float(Config["runtime_projection_safety_factor"])
    ConservativeRate = MeasuredRate * SafetyFactor
    PerSeedUnits = TrainingSamplesPerSeed + ValidationSegments
    PerSeedSeconds = PerSeedUnits / ConservativeRate
    ThreeSeedHours = 3 * PerSeedSeconds / 3600.0
    WallGate = float(Config["full_training_wall_gate_hours"])
    Ready = CacheGate["status"] == "PASS" and ThreeSeedHours <= WallGate
    return {
        "status": "READY_FOR_FULL_TRAINING" if Ready else "BLOCKED_RESOURCE",
        "cache_integrity_status": CacheGate["status"],
        "samples_per_epoch": SamplesPerEpoch,
        "epochs": int(Config["epochs"]),
        "seed_count": len(Config["seeds"]),
        "training_samples_per_seed": TrainingSamplesPerSeed,
        "training_samples_three_seeds": TrainingSamplesPerSeed * len(Config["seeds"]),
        "validation_segments_per_seed": ValidationSegments,
        "measured_cached_end_to_end_samples_per_second": MeasuredRate,
        "projection_safety_factor": SafetyFactor,
        "conservative_samples_per_second": ConservativeRate,
        "projected_hours_per_seed_including_validation": PerSeedSeconds / 3600.0,
        "projected_three_seed_wall_hours": ThreeSeedHours,
        "full_training_wall_gate_hours": WallGate,
        "arbitrary_three_gpu_hour_gate_removed": True,
        "statistical_workload_reduced": False,
    }
