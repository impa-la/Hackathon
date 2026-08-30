# /// <summary>
# Unit and deterministic tiny-smoke checks for the blocked E01 implementation
# /// </summary>

from __future__ import annotations

import io
import hashlib
import json
from dataclasses import replace
import subprocess
import sys
import tempfile
import time
import wave
import zipfile
from pathlib import Path

import numpy as np
import torch


DeepvoiceRoot = Path(__file__).resolve().parents[2]
if str(DeepvoiceRoot) not in sys.path:
    sys.path.insert(0, str(DeepvoiceRoot))

from .audio import (  # noqa: E402
    CalculateSegmentStarts,
    CloseAudioContainerCaches,
    CreateValidationSegments,
    CreateValidationSegmentsWithLengths,
    LoadLocatorWaveform,
    ParseLocator,
)
from .contract_adapter import (  # noqa: E402
    ExpectedE00ContractSha256,
    ObservedE00ContractSha256,
)
from .determinism import ConfigureParentCpuThreads  # noqa: E402
from .benchmark import BenchmarkWindowsWorkers  # noqa: E402
from .cache import CachePaths, ResolveAimeManifestRow  # noqa: E402
from .model import CountTrainableParameters, LogMelCnn  # noqa: E402
from .numerical import (  # noqa: E402
    GuardedFp32OptimizationStep,
    NumericalIntegrityError,
    RequireFiniteTensor,
    RequireFp32TrainingMode,
)
from .records import AudioRecord  # noqa: E402
from .sampling import GroupFirstBalancedSampler  # noqa: E402
from .train_e01 import CalculateMaskedBalancedLoss  # noqa: E402
from .strict_serialization import JsonBytes, NonFinitePayloadError  # noqa: E402


def MakePcmWaveBytes(
    SampleRate: int = 16000,
    Seconds: float = 0.25,
) -> bytes:
    SampleCount = round(SampleRate * Seconds)
    TimeValues = np.arange(SampleCount, dtype=np.float64) / SampleRate
    Samples = np.round(np.sin(2.0 * np.pi * 440.0 * TimeValues) * 10000.0).astype("<i2")
    Buffer = io.BytesIO()
    with wave.open(Buffer, "wb") as WaveFile:
        WaveFile.setnchannels(1)
        WaveFile.setsampwidth(2)
        WaveFile.setframerate(SampleRate)
        WaveFile.writeframes(Samples.tobytes())
    return Buffer.getvalue()


def MakeRecord(
    Dataset: str,
    SampleId: str,
    Group: str,
    SourceFamily: str,
) -> AudioRecord:
    return AudioRecord(
        Dataset=Dataset,
        SampleId=SampleId,
        SourceFamily=SourceFamily,
        GeneratorOrProvider="fixture",
        ContentGroupKey=Group,
        Split="train",
        Locator="forbidden-as-feature",
        Codec="forbidden-as-feature",
        SampleRateHz="forbidden-as-feature",
        Channels="forbidden-as-feature",
        DurationSeconds=8.0,
        Labels=(0.0, 0.0, float("nan"), 1.0, 0.0),
        Masks=(True, True, False, True, True),
    )


def CheckPinnedContract() -> None:
    assert ObservedE00ContractSha256 == ExpectedE00ContractSha256


def CheckLocatorParsing() -> None:
    FileLocator = ParseLocator(r"C:\audio\sample.wav")
    ZipLocator = ParseLocator(r"zip://C:\audio\archive.zip!/folder/sample.wav")
    ParquetLocator = ParseLocator(r"parquet://C:\audio\part.parquet#row=7")
    assert FileLocator.Kind == "file"
    assert ZipLocator.Kind == "zip" and ZipLocator.Member == "folder/sample.wav"
    assert ParquetLocator.Kind == "parquet" and ParquetLocator.RowIndex == 7


def CheckFileZipAndMp3Loaders() -> None:
    WaveBytes = MakePcmWaveBytes()
    try:
        with tempfile.TemporaryDirectory() as TemporaryDirectory:
            Root = Path(TemporaryDirectory)
            WavePath = Root / "sample.wav"
            ZipPath = Root / "sample.zip"
            Mp3Path = Root / "sample.mp3"
            WavePath.write_bytes(WaveBytes)
            with zipfile.ZipFile(ZipPath, "w", compression=zipfile.ZIP_STORED) as ZipFile:
                ZipFile.writestr("sample.wav", WaveBytes)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(WavePath),
                    str(Mp3Path),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            Waveform = LoadLocatorWaveform(str(WavePath), 16000)
            ZipWaveform = LoadLocatorWaveform(f"zip://{ZipPath}!/sample.wav", 16000)
            Mp3Waveform = LoadLocatorWaveform(str(Mp3Path), 16000)
            assert Waveform.numel() == ZipWaveform.numel() == Mp3Waveform.numel() == 4000
            assert torch.allclose(Waveform, ZipWaveform, atol=0.0, rtol=0.0)
            CloseAudioContainerCaches()
    finally:
        CloseAudioContainerCaches()


def CheckParquetDependencyGuard() -> None:
    try:
        LoadLocatorWaveform(r"parquet://C:\missing.parquet#row=0", 16000)
    except FileNotFoundError:
        return
    raise AssertionError("Missing Parquet file did not trigger a guard")


def CheckFixedSegmentContract() -> None:
    SegmentLength = 16000 * 8
    Starts = CalculateSegmentStarts(16000 * 60, SegmentLength, 8)
    assert len(Starts) == 8
    assert Starts[0] == 0
    assert Starts[-1] == 16000 * 60 - SegmentLength
    ShortSegments = CreateValidationSegments(torch.ones(16000 * 4), 16000, 8.0, 8)
    assert ShortSegments.shape == (1, SegmentLength)
    assert float(ShortSegments[0, 16000 * 4 :].abs().sum()) == 0.0


def CheckGroupFirstBalance() -> None:
    Records = [
        MakeRecord("ljspeech-1.1", "sr", "speech-pair", "ljspeech"),
        MakeRecord("wavefake-1.2.0", "sf", "speech-pair", "ljspeech"),
        MakeRecord("fma-small", "mr", "mr", "fma_music"),
        MakeRecord("aime-open-model-subset", "mf", "mf", "aime_music"),
        MakeRecord("wavefake-1.2.0", "held", "held", "jsut_basic5000"),
    ]
    Sampler = GroupFirstBalancedSampler(Records, 400, 20260830)
    Indices = list(iter(Sampler))
    Counts = {Index: Indices.count(Index) for Index in range(len(Records))}
    assert Counts == {0: 100, 1: 100, 2: 100, 3: 100, 4: 0}
    Sampler.SetEpoch(0)
    assert Indices == list(iter(Sampler))


def CheckMaskedLoss() -> None:
    Logits = torch.zeros((4, 5), requires_grad=True)
    Labels = torch.tensor(
        (
            (0, 0, float("nan"), 1, 0),
            (1, 1, float("nan"), 1, 0),
            (0, float("nan"), 0, float("nan"), 1),
            (1, float("nan"), 1, 0, 1),
        ),
        dtype=torch.float32,
    )
    Masks = torch.isfinite(Labels)
    Loss = CalculateMaskedBalancedLoss(Logits, Labels, Masks)
    assert torch.isfinite(Loss)
    Loss.backward()
    assert torch.isfinite(Logits.grad).all()


def CheckMaskedNanLogitAdversary() -> None:
    Logits = torch.zeros((4, 5), requires_grad=True)
    Labels = torch.tensor(
        (
            (0, 0, float("nan"), 1, 0),
            (1, 1, float("nan"), 1, 0),
            (0, float("nan"), 0, float("nan"), 1),
            (1, float("nan"), 1, 0, 1),
        ),
        dtype=torch.float32,
    )
    Masks = torch.isfinite(Labels)
    with torch.no_grad():
        Logits[0, 2] = float("nan")
    Loss = CalculateMaskedBalancedLoss(Logits, Labels, Masks)
    assert torch.isfinite(Loss), "Adversary must prove loss alone can hide a masked NaN"
    try:
        RequireFiniteTensor(Logits, "masked_nan_adversary", 7, "logits")
    except NumericalIntegrityError as Error:
        assert Error.Evidence["stage"] == "masked_nan_adversary"
        assert Error.Evidence["batch_index"] == 7
        assert Error.Evidence["nonfinite_count"] == 1
        return
    raise AssertionError("Masked NaN logit escaped the hard finite guard")


def CheckStrictNestedJsonRejectsNan() -> None:
    try:
        JsonBytes({"outer": [{"inner": float("nan")}]} )
    except NonFinitePayloadError as Error:
        assert "$.outer[0].inner" in str(Error)
        return
    raise AssertionError("Nested NaN was serialized")


def CheckSkippedOptimizerStepRejected() -> None:
    Model = torch.nn.Linear(2, 1)
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-3)
    Optimizer.step = lambda *Arguments, **Keywords: None  # type: ignore[method-assign]
    Inputs = torch.ones((2, 2))
    Logits = Model(Inputs)
    Loss = Logits.square().mean()
    Optimizer.zero_grad(set_to_none=True)
    try:
        GuardedFp32OptimizationStep(
            Model,
            Optimizer,
            Logits,
            Loss,
            "skipped_step_adversary",
            3,
        )
    except NumericalIntegrityError as Error:
        assert Error.Evidence["tensor_name"] == "optimizer_step_behavior"
        assert Error.Evidence["batch_index"] == 3
        return
    raise AssertionError("A silently skipped optimizer step was accepted")


def CheckFp32TrainingMode(Config: dict[str, object]) -> None:
    Evidence = RequireFp32TrainingMode(Config)
    assert Evidence == {
        "precision_mode": "fp32_guarded",
        "autocast_enabled": False,
        "grad_scaler_enabled": False,
    }
    InvalidConfig = dict(Config)
    InvalidConfig["precision_mode"] = "amp_fp16"
    try:
        RequireFp32TrainingMode(InvalidConfig)
    except RuntimeError as Error:
        assert "forbids AMP/GradScaler" in str(Error)
        return
    raise AssertionError("The retired AMP training path was accepted")


def CheckAimeOneBasedRowResolver() -> None:
    import pyarrow as Arrow
    import pyarrow.parquet as Parquet

    WaveBytes = MakePcmWaveBytes()
    with tempfile.TemporaryDirectory() as TemporaryDirectory:
        ParquetPath = Path(TemporaryDirectory) / "aime.parquet"
        AudioType = Arrow.struct(
            (("bytes", Arrow.binary()), ("path", Arrow.string()))
        )
        Table = Arrow.table(
            {
                "id": Arrow.array(["00001", "00002"]),
                "audio": Arrow.array(
                    [
                        {"bytes": WaveBytes, "path": "one.wav"},
                        {"bytes": WaveBytes, "path": "two.wav"},
                    ],
                    type=AudioType,
                ),
            }
        )
        Parquet.write_table(Table, ParquetPath)
        Record = replace(
            MakeRecord("aime-open-model-subset", "00002", "mf", "aime_music"),
            Locator=f"parquet://{ParquetPath}#row=2",
            Labels=(1.0, float("nan"), 1.0, 0.0, 1.0),
            Masks=(True, False, True, True, True),
        )
        Resolution = ResolveAimeManifestRow(Record)
        assert Resolution["resolver_version"] == "aime_manifest_one_based_v1"
        assert Resolution["declared_manifest_row"] == 2
        assert Resolution["resolved_zero_based_row"] == 1
        assert Resolution["asserted_parquet_id"] == "00002"


def CheckWindowsWorkerTaggedDeterminism(Config: dict[str, object]) -> None:
    WaveBytes = MakePcmWaveBytes(Seconds=8.0)
    try:
        with tempfile.TemporaryDirectory() as TemporaryDirectory:
            Root = Path(TemporaryDirectory)
            CacheRoot = Root / "cache"
            WavePath = Root / "real.wav"
            ZipPath = Root / "fake.zip"
            WavePath.write_bytes(WaveBytes)
            with zipfile.ZipFile(ZipPath, "w", compression=zipfile.ZIP_STORED) as ZipFile:
                ZipFile.writestr("fake.wav", WaveBytes)
            SpeechReal = replace(
                MakeRecord("ljspeech-1.1", "sr", "speech-pair", "ljspeech"),
                Locator=str(WavePath),
            )
            SpeechFake = replace(
                MakeRecord("wavefake-1.2.0", "sf", "speech-pair", "ljspeech"),
                Locator=f"zip://{ZipPath}!/fake.wav",
                Labels=(1.0, 1.0, float("nan"), 1.0, 0.0),
            )
            MusicReal = replace(
                MakeRecord("fma-small", "mr", "mr", "fma_music"),
                Locator="cache-only-fma-fixture",
                Labels=(0.0, float("nan"), 0.0, float("nan"), 1.0),
                Masks=(True, False, True, False, True),
            )
            MusicFake = replace(
                MakeRecord("aime-open-model-subset", "mf", "mf", "aime_music"),
                Locator="parquet://cache-only-aime-fixture#row=0",
                Labels=(1.0, float("nan"), 1.0, 0.0, 1.0),
                Masks=(True, False, True, True, True),
            )
            Records = [SpeechReal, SpeechFake, MusicReal, MusicFake]
            CacheArray = np.frombuffer(WaveBytes[44:], dtype="<i2").astype(np.float32) / 32768.0
            for Record in (MusicReal, MusicFake):
                NpyPath, MetadataPath = CachePaths(Record, CacheRoot)
                NpyPath.parent.mkdir(parents=True, exist_ok=True)
                with NpyPath.open("wb") as FileHandle:
                    np.save(FileHandle, CacheArray, allow_pickle=False)
                MetadataPath.write_text("{}", encoding="utf-8")
            WorkerConfig = dict(Config)
            WorkerConfig.update(
                {
                    "resolved_cache_root": str(CacheRoot),
                    "worker_candidates": [0, 2, 4],
                    "worker_benchmark_samples": 16,
                    "batch_size": 4,
                }
            )
            Result = BenchmarkWindowsWorkers(Records, WorkerConfig)
            assert Result["status"] == "PASS"
            assert Result["cross_worker_tensor_sequence_exact"] is True
            assert Result["canonical_masked_label_storage"] is True
            assert Result["parent_intraop_threads"] == 1
            assert Result["parent_interop_threads"] == 1
            assert Result["worker_intraop_threads"] == 1
            assert Result["warm_tagged_epoch"] == 0
            assert Result["measured_tagged_epoch"] == 1
            CloseAudioContainerCaches()
    finally:
        CloseAudioContainerCaches()


def CheckReferenceModelContract(Config: dict[str, object]) -> None:
    torch.manual_seed(20260830)
    Model = LogMelCnn(Config)
    Model.eval()
    Waveforms = torch.zeros((2, int(Config["sample_rate"]) * 8))
    First = Model(Waveforms)
    Second = Model(Waveforms)
    assert First.shape == (2, 5)
    assert torch.allclose(First, Second, atol=0.0, rtol=0.0)
    assert CountTrainableParameters(Model) > 0


def CheckPaddingMaskContract(Config: dict[str, object]) -> None:
    torch.manual_seed(20260830)
    Model = LogMelCnn(Config).eval()
    SegmentLength = int(Config["sample_rate"]) * 8
    ValidLength = int(Config["sample_rate"]) * 4
    First = torch.zeros((1, SegmentLength))
    First[:, :ValidLength] = torch.randn((1, ValidLength)) * 0.01
    Second = First.clone()
    Second[:, ValidLength:] = 100.0
    Counts = torch.tensor([ValidLength])
    with torch.inference_mode():
        FirstPrediction = Model(First, Counts)
        SecondPrediction = Model(Second, Counts)
    assert torch.allclose(FirstPrediction, SecondPrediction, atol=0.0, rtol=0.0)
    Segments, Lengths = CreateValidationSegmentsWithLengths(
        First[0, :ValidLength], 16000, 8.0, 8
    )
    assert Segments.shape == (1, SegmentLength)
    assert Lengths.tolist() == [ValidLength]


def RunTinySmoke(
    Config: dict[str, object],
    DeviceName: str = "cpu",
) -> dict[str, object]:
    Started = time.perf_counter()
    torch.manual_seed(20260830)
    Device = torch.device(DeviceName)
    PrecisionEvidence = RequireFp32TrainingMode(Config)
    Model = LogMelCnn(Config).to(Device)
    Model.train()
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-4)
    Waveforms = (
        torch.randn((4, int(Config["sample_rate"]) * 8), device=Device) * 0.01
    )
    Labels = torch.tensor(
        (
            (0, 0, float("nan"), 1, 0),
            (1, 1, float("nan"), 1, 0),
            (0, float("nan"), 0, float("nan"), 1),
            (1, float("nan"), 1, 0, 1),
        ),
        dtype=torch.float32,
    )
    Labels = Labels.to(Device)
    Masks = torch.isfinite(Labels)
    ValidSampleCounts = torch.full(
        (Waveforms.shape[0],),
        Waveforms.shape[1],
        dtype=torch.long,
        device=Device,
    )
    if Device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(Device)
    torch.cuda.synchronize() if Device.type == "cuda" else None
    ThroughputStarted = time.perf_counter()
    Optimizer.zero_grad(set_to_none=True)
    Logits = Model(Waveforms, ValidSampleCounts)
    Loss = CalculateMaskedBalancedLoss(Logits, Labels, Masks)
    StepEvidence = GuardedFp32OptimizationStep(
        Model,
        Optimizer,
        Logits,
        Loss,
        f"tiny_{Device.type}_smoke",
        0,
    )
    torch.cuda.synchronize() if Device.type == "cuda" else None
    TrainingSeconds = time.perf_counter() - ThroughputStarted
    Model.eval()
    with torch.inference_mode():
        Predictions = torch.sigmoid(Model(Waveforms, ValidSampleCounts)).cpu()
    RequireFiniteTensor(Predictions, "tiny_smoke", 0, "predictions")
    Digest = hashlib.sha256(Predictions.numpy().tobytes()).hexdigest()
    return {
        "status": "PASS",
        "scope": "synthetic one-step smoke only",
        "is_e01_performance_result": False,
        "seed": 20260830,
        "device": str(Device),
        "batch_size": 4,
        "segment_seconds": 8.0,
        "loss": float(Loss.detach()),
        "prediction_sha256": Digest,
        "trainable_parameters": CountTrainableParameters(Model),
        "seconds": time.perf_counter() - Started,
        "training_step_seconds": TrainingSeconds,
        "training_segments_per_second": Waveforms.shape[0] / TrainingSeconds,
        **PrecisionEvidence,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(Device) if Device.type == "cuda" else None
        ),
        "optimizer_step_before": StepEvidence["optimizer_step_before"],
        "optimizer_step_after": StepEvidence["optimizer_step_after"],
    }


def RunAllTests(Config: dict[str, object]) -> dict[str, object]:
    Checks = (
        CheckPinnedContract,
        CheckLocatorParsing,
        CheckFileZipAndMp3Loaders,
        CheckParquetDependencyGuard,
        CheckFixedSegmentContract,
        CheckGroupFirstBalance,
        CheckMaskedLoss,
        CheckMaskedNanLogitAdversary,
        CheckStrictNestedJsonRejectsNan,
        CheckSkippedOptimizerStepRejected,
        lambda: CheckFp32TrainingMode(Config),
        CheckAimeOneBasedRowResolver,
        lambda: CheckReferenceModelContract(Config),
        lambda: CheckPaddingMaskContract(Config),
        lambda: CheckWindowsWorkerTaggedDeterminism(Config),
    )
    Results = []
    for Check in Checks:
        Check()
        Results.append({"check": getattr(Check, "__name__", "CheckReferenceModelContract"), "status": "PASS"})
    return {"status": "PASS", "check_count": len(Results), "checks": Results}


def Main() -> int:
    ConfigPath = Path(__file__).resolve().parent / "config.json"
    Config = json.loads(ConfigPath.read_text(encoding="utf-8"))
    ConfigureParentCpuThreads(Config)
    print(json.dumps(RunAllTests(Config), indent=2, ensure_ascii=False))
    print(json.dumps(RunTinySmoke(Config), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
