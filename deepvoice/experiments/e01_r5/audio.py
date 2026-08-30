# /// <summary>
# Waveform-only audio loading and fixed E01 segmentation for file, ZIP and Parquet locators
# /// </summary>

from __future__ import annotations

import io
import functools
import gc
import math
import re
import subprocess
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ParsedLocator:
    Kind: str
    ContainerPath: Path
    Member: str | None = None
    RowIndex: int | None = None


def ParseLocator(Locator: str) -> ParsedLocator:
    if Locator.startswith("zip://"):
        Payload = Locator[len("zip://") :]
        if "!/" not in Payload:
            raise ValueError("ZIP locator must contain !/")
        ContainerText, Member = Payload.split("!/", 1)
        if not ContainerText or not Member:
            raise ValueError("ZIP locator has an empty container or member")
        return ParsedLocator("zip", Path(ContainerText), Member=Member)
    if Locator.startswith("parquet://"):
        Payload = Locator[len("parquet://") :]
        Match = re.fullmatch(r"(.+)#row=(\d+)", Payload)
        if Match is None:
            raise ValueError("Parquet locator must end with #row=<nonnegative integer>")
        return ParsedLocator(
            "parquet",
            Path(Match.group(1)),
            RowIndex=int(Match.group(2)),
        )
    return ParsedLocator("file", Path(Locator))


def DecodePcmWaveBytes(AudioBytes: bytes) -> tuple[torch.Tensor, int]:
    with wave.open(io.BytesIO(AudioBytes), "rb") as WaveFile:
        ChannelCount = WaveFile.getnchannels()
        SampleWidth = WaveFile.getsampwidth()
        SampleRate = WaveFile.getframerate()
        FrameCount = WaveFile.getnframes()
        RawSamples = WaveFile.readframes(FrameCount)
    if SampleWidth != 2:
        raise ValueError("PCM fallback supports 16-bit WAV only")
    Samples = np.frombuffer(RawSamples, dtype="<i2").astype(np.float32) / 32768.0
    Samples = Samples.reshape(-1, ChannelCount).T
    return torch.from_numpy(Samples.copy()), int(SampleRate)


def DecodePcmWavePath(AudioPath: Path) -> tuple[torch.Tensor, int]:
    return DecodePcmWaveBytes(AudioPath.read_bytes())


def DecodeWithTorchaudioPath(AudioPath: Path) -> tuple[torch.Tensor, int]:
    import torchaudio

    Waveform, SampleRate = torchaudio.load(str(AudioPath))
    return Waveform.float(), int(SampleRate)


def DecodeWithTorchaudioBytes(AudioBytes: bytes) -> tuple[torch.Tensor, int]:
    import torchaudio

    Waveform, SampleRate = torchaudio.load(io.BytesIO(AudioBytes))
    return Waveform.float(), int(SampleRate)


def DecodeWithFfmpeg(
    TargetSampleRate: int,
    AudioPath: Path | None = None,
    AudioBytes: bytes | None = None,
) -> torch.Tensor:
    if (AudioPath is None) == (AudioBytes is None):
        raise ValueError("Exactly one ffmpeg input must be provided")
    InputName = "pipe:0" if AudioBytes is not None else str(AudioPath)
    Command = [
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
        str(TargetSampleRate),
        "pipe:1",
    ]
    Result = subprocess.run(
        Command,
        input=AudioBytes,
        check=True,
        capture_output=True,
        timeout=180,
    )
    Samples = np.frombuffer(Result.stdout, dtype="<f4").copy()
    if Samples.size == 0:
        raise ValueError("ffmpeg returned empty audio")
    return torch.from_numpy(Samples)


def ResampleWaveform(
    Waveform: torch.Tensor,
    SourceSampleRate: int,
    TargetSampleRate: int,
) -> torch.Tensor:
    if SourceSampleRate == TargetSampleRate:
        return Waveform
    TargetLength = max(
        1,
        round(Waveform.shape[-1] * TargetSampleRate / SourceSampleRate),
    )
    return F.interpolate(
        Waveform.unsqueeze(0),
        size=TargetLength,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def FinalizeWaveform(
    Waveform: torch.Tensor,
    SourceSampleRate: int,
    TargetSampleRate: int,
) -> torch.Tensor:
    if Waveform.ndim == 1:
        Waveform = Waveform.unsqueeze(0)
    if Waveform.ndim != 2 or Waveform.numel() == 0:
        raise ValueError("Decoded waveform must be nonempty channels by time")
    if not torch.isfinite(Waveform).all():
        raise ValueError("Decoded waveform contains nonfinite values")
    MonoWaveform = Waveform.float().mean(dim=0, keepdim=True)
    MonoWaveform = ResampleWaveform(
        MonoWaveform,
        SourceSampleRate,
        TargetSampleRate,
    )
    return MonoWaveform.squeeze(0).clamp(-1.0, 1.0)


@functools.lru_cache(maxsize=36)
def LoadParquetAudioColumn(ParquetPathText: str):
    try:
        import pyarrow.parquet as Parquet
    except ImportError as Error:
        raise RuntimeError(
            "pyarrow is required for parquet:// AIME locators"
        ) from Error
    return Parquet.read_table(Path(ParquetPathText), columns=["audio"]).column("audio")


def LoadParquetAudioBytes(ParquetPath: Path, RowIndex: int) -> bytes:
    AudioColumn = LoadParquetAudioColumn(str(ParquetPath))
    RowCount = len(AudioColumn)
    if not 0 <= RowIndex < RowCount:
        raise IndexError(f"Parquet row {RowIndex} is outside {RowCount} rows")
    AudioValue = AudioColumn[RowIndex].as_py()
    if not isinstance(AudioValue, dict) or not isinstance(AudioValue.get("bytes"), bytes):
        raise TypeError("AIME audio must be a struct containing embedded bytes")
    return AudioValue["bytes"]


@functools.lru_cache(maxsize=4)
def OpenZipArchive(ArchivePathText: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(Path(ArchivePathText), "r")


def CloseAudioContainerCaches() -> None:
    OpenZipArchive.cache_clear()
    LoadParquetAudioColumn.cache_clear()
    gc.collect()


def LoadLocatorWaveform(
    Locator: str,
    TargetSampleRate: int = 16000,
    Diagnostics: dict[str, object] | None = None,
) -> torch.Tensor:
    Parsed = ParseLocator(Locator)
    if not Parsed.ContainerPath.is_file():
        raise FileNotFoundError(f"Audio container not found: {Parsed.ContainerPath}")

    if Parsed.Kind == "file":
        try:
            Waveform, SampleRate = DecodeWithTorchaudioPath(Parsed.ContainerPath)
            if Diagnostics is not None:
                Diagnostics["decoder_backend"] = "torchaudio_path"
            return FinalizeWaveform(Waveform, SampleRate, TargetSampleRate)
        except Exception:
            pass
        if Parsed.ContainerPath.suffix.casefold() == ".wav":
            try:
                Waveform, SampleRate = DecodePcmWavePath(Parsed.ContainerPath)
                if Diagnostics is not None:
                    Diagnostics["decoder_backend"] = "stdlib_pcm_path"
                return FinalizeWaveform(Waveform, SampleRate, TargetSampleRate)
            except Exception:
                pass
        if Diagnostics is not None:
            Diagnostics["decoder_backend"] = "ffmpeg_subprocess_path"
        return DecodeWithFfmpeg(TargetSampleRate, AudioPath=Parsed.ContainerPath)

    if Parsed.Kind == "zip":
        if Parsed.Member is None:
            raise AssertionError("ZIP member was not parsed")
        AudioBytes = OpenZipArchive(str(Parsed.ContainerPath)).read(Parsed.Member)
        try:
            Waveform, SampleRate = DecodeWithTorchaudioBytes(AudioBytes)
            if Diagnostics is not None:
                Diagnostics["decoder_backend"] = "torchaudio_bytes"
            return FinalizeWaveform(Waveform, SampleRate, TargetSampleRate)
        except Exception:
            pass
        try:
            Waveform, SampleRate = DecodePcmWaveBytes(AudioBytes)
            if Diagnostics is not None:
                Diagnostics["decoder_backend"] = "stdlib_pcm_bytes"
            return FinalizeWaveform(Waveform, SampleRate, TargetSampleRate)
        except Exception:
            if Diagnostics is not None:
                Diagnostics["decoder_backend"] = "ffmpeg_subprocess_bytes"
            return DecodeWithFfmpeg(TargetSampleRate, AudioBytes=AudioBytes)

    if Parsed.Kind == "parquet":
        if Parsed.RowIndex is None:
            raise AssertionError("Parquet row was not parsed")
        AudioBytes = LoadParquetAudioBytes(Parsed.ContainerPath, Parsed.RowIndex)
        try:
            Waveform, SampleRate = DecodeWithTorchaudioBytes(AudioBytes)
            if Diagnostics is not None:
                Diagnostics["decoder_backend"] = "torchaudio_bytes"
            return FinalizeWaveform(Waveform, SampleRate, TargetSampleRate)
        except Exception:
            pass
        try:
            Waveform, SampleRate = DecodePcmWaveBytes(AudioBytes)
            if Diagnostics is not None:
                Diagnostics["decoder_backend"] = "stdlib_pcm_bytes"
            return FinalizeWaveform(Waveform, SampleRate, TargetSampleRate)
        except Exception:
            if Diagnostics is not None:
                Diagnostics["decoder_backend"] = "ffmpeg_subprocess_bytes"
            return DecodeWithFfmpeg(TargetSampleRate, AudioBytes=AudioBytes)

    raise AssertionError(f"Unsupported parsed locator kind: {Parsed.Kind}")


def CalculateSegmentStarts(
    SampleCount: int,
    SegmentLength: int,
    MaximumSegments: int,
) -> list[int]:
    if SampleCount <= 0 or SegmentLength <= 0 or MaximumSegments <= 0:
        raise ValueError("Segment dimensions must be positive")
    if SampleCount <= SegmentLength:
        return [0]
    LastStart = SampleCount - SegmentLength
    NaturalSegmentCount = math.ceil(SampleCount / SegmentLength)
    SegmentCount = min(MaximumSegments, max(2, NaturalSegmentCount))
    Starts = np.linspace(0, LastStart, SegmentCount, dtype=np.float64)
    RoundedStarts = [int(round(Value)) for Value in Starts]
    return list(dict.fromkeys(RoundedStarts))


def CreateValidationSegmentsWithLengths(
    Waveform: torch.Tensor,
    SampleRate: int = 16000,
    SegmentSeconds: float = 8.0,
    MaximumSegments: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    SegmentLength = round(SampleRate * SegmentSeconds)
    Starts = CalculateSegmentStarts(
        Waveform.numel(),
        SegmentLength,
        MaximumSegments,
    )
    Segments = []
    ValidSampleCounts = []
    for Start in Starts:
        Segment = Waveform[Start : Start + SegmentLength]
        ValidSampleCounts.append(min(Segment.numel(), SegmentLength))
        if Segment.numel() < SegmentLength:
            Segment = F.pad(Segment, (0, SegmentLength - Segment.numel()))
        Segments.append(Segment)
    return torch.stack(Segments), torch.tensor(ValidSampleCounts, dtype=torch.long)


def CreateValidationSegments(
    Waveform: torch.Tensor,
    SampleRate: int = 16000,
    SegmentSeconds: float = 8.0,
    MaximumSegments: int = 8,
) -> torch.Tensor:
    Segments, _ = CreateValidationSegmentsWithLengths(
        Waveform,
        SampleRate,
        SegmentSeconds,
        MaximumSegments,
    )
    return Segments


def CreateTrainingSegmentWithLength(
    Waveform: torch.Tensor,
    DeterministicOffset: float,
    SampleRate: int = 16000,
    SegmentSeconds: float = 8.0,
) -> tuple[torch.Tensor, int]:
    if not 0.0 <= DeterministicOffset <= 1.0:
        raise ValueError("Training offset must be in [0, 1]")
    SegmentLength = round(SampleRate * SegmentSeconds)
    if Waveform.numel() <= SegmentLength:
        ValidSampleCount = Waveform.numel()
        return (
            F.pad(Waveform, (0, SegmentLength - Waveform.numel())),
            ValidSampleCount,
        )
    LastStart = Waveform.numel() - SegmentLength
    Start = int(round(DeterministicOffset * LastStart))
    return Waveform[Start : Start + SegmentLength], SegmentLength


def CreateTrainingSegment(
    Waveform: torch.Tensor,
    DeterministicOffset: float,
    SampleRate: int = 16000,
    SegmentSeconds: float = 8.0,
) -> torch.Tensor:
    Segment, _ = CreateTrainingSegmentWithLength(
        Waveform,
        DeterministicOffset,
        SampleRate,
        SegmentSeconds,
    )
    return Segment
