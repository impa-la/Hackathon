# /// <summary>
# GPU batch autotune and measured E01 throughput projection
# /// </summary>

from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import numpy as np
import torch

from .audio import CreateTrainingSegmentWithLength, LoadLocatorWaveform
from .model import LogMelCnn
from .records import AudioRecord
from .sampling import ClassifyTrainingStratum, GroupFirstBalancedSampler
from .train_e01 import CalculateMaskedBalancedLoss


def BenchmarkGpuBatch(Config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "CUDA unavailable", "candidates": []}
    Device = torch.device("cuda")
    CandidateRows = []
    SegmentLength = round(
        int(Config["sample_rate"]) * float(Config["segment_seconds"])
    )
    TotalMemory = torch.cuda.get_device_properties(0).total_memory
    for BatchSize in (4, 8, 12, 16, 24, 32):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.manual_seed(20260830)
            Model = LogMelCnn(Config).to(Device).train()
            Optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-4)
            Waveforms = torch.randn(
                BatchSize,
                SegmentLength,
                device=Device,
                dtype=torch.float32,
            ) * 0.01
            ValidSampleCounts = torch.full(
                (BatchSize,), SegmentLength, device=Device, dtype=torch.long
            )
            for _ in range(1):
                Optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    Loss = Model(Waveforms, ValidSampleCounts).square().mean()
                Loss.backward()
                Optimizer.step()
            torch.cuda.synchronize()
            Started = time.perf_counter()
            Iterations = 3
            for _ in range(Iterations):
                Optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    Loss = Model(Waveforms, ValidSampleCounts).square().mean()
                Loss.backward()
                Optimizer.step()
            torch.cuda.synchronize()
            Seconds = time.perf_counter() - Started
            PeakBytes = torch.cuda.max_memory_allocated()
            Row = {
                "batch_size": BatchSize,
                "status": "PASS",
                "iterations": Iterations,
                "seconds": Seconds,
                "segments_per_second": BatchSize * Iterations / Seconds,
                "peak_allocated_bytes": PeakBytes,
                "peak_allocated_gib": PeakBytes / 1024**3,
                "memory_fraction": PeakBytes / TotalMemory,
            }
            CandidateRows.append(Row)
            del Model, Optimizer, Waveforms, ValidSampleCounts, Loss
        except torch.cuda.OutOfMemoryError as Error:
            CandidateRows.append(
                {
                    "batch_size": BatchSize,
                    "status": "OOM",
                    "error": str(Error),
                }
            )
            torch.cuda.empty_cache()
            break
    PassingRows = [Row for Row in CandidateRows if Row["status"] == "PASS"]
    EligibleRows = [
        Row for Row in PassingRows if Row["memory_fraction"] <= 0.85
    ]
    if not EligibleRows:
        return {
            "status": "BLOCKED",
            "reason": "No batch passed the 85% VRAM gate",
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
        "selection_rule": "highest measured segments/s among batches using <=85% VRAM",
        "recommended_batch_size": RecommendedRow["batch_size"],
        "recommended_segments_per_second": RecommendedRow["segments_per_second"],
        "recommended_peak_allocated_bytes": RecommendedRow["peak_allocated_bytes"],
        "candidates": CandidateRows,
    }


def SelectProbeRecords(
    TrainingRecords: Sequence[AudioRecord],
) -> dict[str, AudioRecord]:
    Probes: dict[str, AudioRecord] = {}
    for Record in TrainingRecords:
        Stratum = ClassifyTrainingStratum(Record)
        if Stratum is not None:
            Probes.setdefault(Stratum, Record)
    return Probes


def BenchmarkLoaders(
    TrainingRecords: Sequence[AudioRecord],
    TargetSampleRate: int,
) -> dict[str, Any]:
    Probes = SelectProbeRecords(TrainingRecords)
    Rows = []
    for Stratum in ("speech_real", "speech_fake", "music_real", "music_fake"):
        Record = Probes[Stratum]
        AttemptSeconds = []
        SampleCounts = []
        for _ in range(2):
            Started = time.perf_counter()
            Diagnostics: dict[str, object] = {}
            Waveform = LoadLocatorWaveform(
                Record.Locator,
                TargetSampleRate,
                Diagnostics,
            )
            AttemptSeconds.append(time.perf_counter() - Started)
            SampleCounts.append(int(Waveform.numel()))
        Rows.append(
            {
                "stratum": Stratum,
                "dataset": Record.Dataset,
                "attempt_seconds": AttemptSeconds,
                "warm_decode_seconds": AttemptSeconds[-1],
                "decoded_sample_count": SampleCounts[-1],
                "decoder_backend": Diagnostics.get("decoder_backend"),
            }
        )
    MeanBalancedDecodeSeconds = float(
        np.mean([Row["warm_decode_seconds"] for Row in Rows])
    )
    return {
        "status": "PASS",
        "probes": Rows,
        "balanced_mean_warm_decode_seconds": MeanBalancedDecodeSeconds,
        "balanced_files_per_second": 1.0 / MeanBalancedDecodeSeconds,
    }


def BenchmarkBalancedPilot(
    TrainingRecords: Sequence[AudioRecord],
    Config: dict[str, Any],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "CUDA unavailable"}
    Seed = 20260830
    PilotSamples = int(Config["pilot_samples"])
    Sampler = GroupFirstBalancedSampler(TrainingRecords, PilotSamples, Seed)
    Indices = list(iter(Sampler))
    Device = torch.device("cuda")
    SegmentLength = round(
        int(Config["sample_rate"]) * float(Config["segment_seconds"])
    )
    DecodeStarted = time.perf_counter()
    Segments = []
    ValidCounts = []
    Labels = []
    Masks = []
    Backends: dict[str, int] = {}
    LocatorDigest = hashlib.sha256()
    for Position, RecordIndex in enumerate(Indices):
        Record = TrainingRecords[RecordIndex]
        Diagnostics: dict[str, object] = {}
        Waveform = LoadLocatorWaveform(
            Record.Locator,
            int(Config["sample_rate"]),
            Diagnostics,
        )
        Backend = str(Diagnostics.get("decoder_backend", "unknown"))
        Backends[Backend] = Backends.get(Backend, 0) + 1
        LocatorDigest.update(Record.Locator.encode("utf-8"))
        OffsetDigest = hashlib.sha256(
            f"{Seed}|pilot|{Position}|{Record.SampleId}".encode("utf-8")
        ).digest()
        Offset = int.from_bytes(OffsetDigest[:8], "big") / float(2**64 - 1)
        Segment, ValidCount = CreateTrainingSegmentWithLength(
            Waveform,
            Offset,
            int(Config["sample_rate"]),
            float(Config["segment_seconds"]),
        )
        Segments.append(Segment)
        ValidCounts.append(ValidCount)
        Labels.append(Record.Labels)
        Masks.append(Record.Masks)
    DecodeSeconds = time.perf_counter() - DecodeStarted
    SegmentTensor = torch.stack(Segments)
    ValidTensor = torch.tensor(ValidCounts, dtype=torch.long)
    LabelTensor = torch.tensor(Labels, dtype=torch.float32)
    MaskTensor = torch.tensor(Masks, dtype=torch.bool)
    Model = LogMelCnn(Config).to(Device).train()
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=float(Config["learning_rate"]))
    Scaler = torch.amp.GradScaler("cuda", enabled=True)
    BatchSize = int(Config["batch_size"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ComputeStarted = time.perf_counter()
    BatchCount = 0
    for Start in range(0, PilotSamples, BatchSize):
        Waveforms = SegmentTensor[Start : Start + BatchSize].to(Device)
        BatchValid = ValidTensor[Start : Start + BatchSize].to(Device)
        BatchLabels = LabelTensor[Start : Start + BatchSize].to(Device)
        BatchMasks = MaskTensor[Start : Start + BatchSize].to(Device)
        Optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            Logits = Model(Waveforms, BatchValid)
            Loss = CalculateMaskedBalancedLoss(Logits, BatchLabels, BatchMasks)
        if not torch.isfinite(Loss):
            raise RuntimeError(f"Balanced pilot produced nonfinite loss at batch {BatchCount}")
        Scaler.scale(Loss).backward()
        Scaler.step(Optimizer)
        Scaler.update()
        BatchCount += 1
    torch.cuda.synchronize()
    ComputeSeconds = time.perf_counter() - ComputeStarted
    PeakBytes = torch.cuda.max_memory_allocated()
    TotalSeconds = DecodeSeconds + ComputeSeconds
    return {
        "status": "PASS",
        "scope": "deterministic balanced real-locator training pilot; not a performance result",
        "is_e01_performance_result": False,
        "seed": Seed,
        "sample_count": PilotSamples,
        "batch_size": BatchSize,
        "batch_count": BatchCount,
        "locator_sequence_sha256": LocatorDigest.hexdigest(),
        "decoder_backend_counts": Backends,
        "decode_seconds": DecodeSeconds,
        "compute_seconds": ComputeSeconds,
        "total_seconds": TotalSeconds,
        "decode_files_per_second": PilotSamples / DecodeSeconds,
        "compute_segments_per_second": PilotSamples / ComputeSeconds,
        "end_to_end_samples_per_second": PilotSamples / TotalSeconds,
        "peak_allocated_bytes": PeakBytes,
        "peak_allocated_gib": PeakBytes / 1024**3,
        "final_loss": float(Loss.detach().cpu()),
    }


def ProjectRuntime(
    Config: dict[str, Any],
    GpuBenchmark: dict[str, Any],
    LoaderBenchmark: dict[str, Any],
    ValidationRecords: Sequence[AudioRecord],
    BalancedPilot: dict[str, Any],
) -> dict[str, Any]:
    SamplesPerEpoch = int(Config["samples_per_epoch"])
    ExpectedSamplesPerEpoch = (
        int(Config["balanced_group_draws_per_epoch"])
        * int(Config["samples_per_balanced_group_draw"])
    )
    if SamplesPerEpoch != ExpectedSamplesPerEpoch:
        raise RuntimeError("E01 samples-per-epoch contract is internally inconsistent")
    TrainingSamplesPerSeed = int(Config["epochs"]) * SamplesPerEpoch
    GpuSegmentsPerSecond = float(GpuBenchmark["recommended_segments_per_second"])
    LoaderFilesPerSecond = float(LoaderBenchmark["balanced_files_per_second"])
    PilotRate = float(BalancedPilot["end_to_end_samples_per_second"])
    EffectiveTrainingRate = min(
        GpuSegmentsPerSecond,
        LoaderFilesPerSecond,
        PilotRate,
    )
    TrainingSecondsPerSeed = TrainingSamplesPerSeed / EffectiveTrainingRate
    ValidationSegments = sum(
        min(
            int(Config["max_segments_per_file"]),
            max(1, int(np.ceil(Record.DurationSeconds / float(Config["segment_seconds"])))),
        )
        for Record in ValidationRecords
    )
    ValidationDecodeSecondsPerSeed = (
        len(ValidationRecords) / LoaderFilesPerSecond
    )
    ValidationGpuSecondsPerSeed = ValidationSegments / GpuSegmentsPerSecond
    ValidationSecondsPerSeed = ValidationDecodeSecondsPerSeed + ValidationGpuSecondsPerSeed
    TotalSeconds = 3 * (TrainingSecondsPerSeed + ValidationSecondsPerSeed)
    Projection = {
        "training_samples_per_seed": TrainingSamplesPerSeed,
        "balanced_group_draws_per_epoch": int(
            Config["balanced_group_draws_per_epoch"]
        ),
        "samples_per_balanced_group_draw": int(
            Config["samples_per_balanced_group_draw"]
        ),
        "samples_per_epoch": SamplesPerEpoch,
        "validation_files_per_seed": len(ValidationRecords),
        "validation_segments_per_seed": ValidationSegments,
        "gpu_segments_per_second": GpuSegmentsPerSecond,
        "loader_files_per_second": LoaderFilesPerSecond,
        "effective_training_samples_per_second": EffectiveTrainingRate,
        "balanced_pilot_samples_per_second": PilotRate,
        "training_hours_per_seed": TrainingSecondsPerSeed / 3600.0,
        "validation_hours_per_seed": ValidationSecondsPerSeed / 3600.0,
        "projected_three_seed_wall_hours": TotalSeconds / 3600.0,
        "full_run_gate_gpu_hours": 3.0,
        "full_run_gate_wall_hours": 24.0,
    }
    Projection["status"] = (
        "READY"
        if Projection["projected_three_seed_wall_hours"] <= 3.0
        else "BLOCKED_RESOURCE"
    )
    return Projection
