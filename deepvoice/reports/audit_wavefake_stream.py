#!/usr/bin/env python3
"""Stream-audit WaveFake 1.2.0 directly inside its official ZIP archive.

The source ZIP is opened read-only and is never extracted, rewritten, or moved. Every
member is read to EOF once so Python's ZIP reader verifies CRC while this script also
computes CRC-32, complete-file SHA-256, PCM SHA-256, WAV structure, and PCM quality.
All outputs are written beneath --output-dir. A completed-row inventory makes the run
resumable without modifying source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import time
import zlib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import numpy as np


ExpectedArchiveBytes = 28_918_626_084
ExpectedArchiveMd5 = "76b3e62d69f866e57ad6b1debaff434b"
ExpectedArchiveSha256 = "431e880dc54361cbb9722f332377f20d4524d231ecfce8ce9ca187dc1b6bec30"
ExpectedMemberCount = 134_266
ExpectedCompressedBytes = 28_878_627_156
ExpectedUncompressedBytes = 34_522_472_424
SilenceThreshold = 1e-4
ClippingThreshold = 0.999
PcmBlockBytes = 1_048_576
ProgressInterval = 250
SplitSalt = "wavefake-content-group-split-v1"

LjsIdPattern = re.compile(r"(LJ\d{3}-\d{4})", re.IGNORECASE)
JsutIdPattern = re.compile(r"(BASIC5000_\d{4})", re.IGNORECASE)
CommonVoiceIdPattern = re.compile(r"gen_(\d+)", re.IGNORECASE)

LjsDirectories = [
    "ljspeech_full_band_melgan",
    "ljspeech_hifiGAN",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]
JsutDirectories = ["jsut_multi_band_melgan", "jsut_parallel_wavegan"]
CommonVoiceDirectory = "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech"

GeneratorByDirectory = {
    "ljspeech_full_band_melgan": "full_band_melgan",
    "ljspeech_hifiGAN": "hifigan",
    "ljspeech_melgan": "melgan",
    "ljspeech_melgan_large": "melgan_large",
    "ljspeech_multi_band_melgan": "multi_band_melgan",
    "ljspeech_parallel_wavegan": "parallel_wavegan",
    "ljspeech_waveglow": "waveglow",
    "jsut_multi_band_melgan": "multi_band_melgan",
    "jsut_parallel_wavegan": "parallel_wavegan",
    CommonVoiceDirectory: "conformer_fastspeech2_parallel_wavegan",
}

InventoryFields = [
    "zip_member",
    "dataset_dir",
    "role",
    "source_family",
    "source_id",
    "generator",
    "content_group_key",
    "recommended_content_split",
    "compressed_bytes",
    "uncompressed_bytes",
    "compression_method",
    "expected_crc32",
    "actual_crc32",
    "crc_status",
    "bytes_read",
    "file_sha256",
    "signature_status",
    "parse_status",
    "parse_error",
    "riff_declared_bytes",
    "audio_format_tag",
    "codec",
    "channels",
    "sample_rate_hz",
    "bits_per_sample",
    "valid_bits_per_sample",
    "block_align",
    "byte_rate",
    "fmt_chunk_bytes",
    "data_bytes",
    "frames",
    "duration_seconds",
    "pcm_sha256",
    "sample_count",
    "nonfinite_sample_count",
    "peak_abs_normalized",
    "rms_normalized",
    "silent_sample_count",
    "silent_sample_fraction_lt_1e_4",
    "clipped_sample_count",
    "clipped_sample_fraction_ge_0_999",
    "license_id",
    "attribution_text",
    "changes_notice_required_when_shared",
    "sharealike_required_when_adapted",
    "training_eligible",
    "eligibility_reason",
    "issues",
]


class TrackingReader:
    def __init__(self, Stream: BinaryIO):
        if Stream is None:
            raise ValueError("Stream is required")
        self.Stream = Stream
        self.FileDigest = hashlib.sha256()
        self.Crc32 = 0
        self.BytesRead = 0

    def Read(self, Size: int = -1) -> bytes:
        Data = self.Stream.read(Size)
        if Data:
            self.FileDigest.update(Data)
            self.Crc32 = zlib.crc32(Data, self.Crc32)
            self.BytesRead += len(Data)
        return Data

    def ReadExact(self, Size: int) -> bytes:
        if Size < 0:
            raise ValueError("Size must not be negative")
        Parts: list[bytes] = []
        Remaining = Size
        while Remaining:
            Part = self.Read(Remaining)
            if not Part:
                raise EOFError(f"Expected {Size} bytes but reached EOF with {Remaining} remaining")
            Parts.append(Part)
            Remaining -= len(Part)
        return b"".join(Parts)

    def Drain(self) -> None:
        while self.Read(8 * 1024 * 1024):
            pass


def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--archive", type=Path, required=True)
    Parser.add_argument("--ljspeech-inventory", type=Path, required=True)
    Parser.add_argument("--download-run", type=Path, required=True)
    Parser.add_argument("--output-dir", type=Path, required=True)
    Parser.add_argument("--resume", action="store_true")
    return Parser.parse_args()


def FileState(PathValue: Path) -> dict[str, int | str]:
    Stat = PathValue.stat()
    return {"path": str(PathValue), "size_bytes": Stat.st_size, "mtime_ns": Stat.st_mtime_ns}


def IsUnsafePath(Name: str) -> bool:
    Normalized = Name.replace("\\", "/")
    Value = PurePosixPath(Normalized)
    return Value.is_absolute() or ".." in Value.parts or bool(re.match(r"^[A-Za-z]:", Normalized))


def IsSymlink(Info: zipfile.ZipInfo) -> bool:
    UnixMode = (Info.external_attr >> 16) & 0o170000
    return UnixMode == 0o120000


def ClassifyMember(Name: str) -> dict[str, str]:
    Parts = Name.replace("\\", "/").split("/")
    Directory = Parts[1] if len(Parts) > 2 else ""
    FileName = Parts[-1]
    Role = "generated"
    SourceFamily = "unknown"
    SourceId = ""
    Generator = GeneratorByDirectory.get(Directory, "unknown")
    if Directory in LjsDirectories:
        SourceFamily = "ljspeech"
        Match = LjsIdPattern.search(FileName)
        SourceId = Match.group(1).upper() if Match else ""
    elif Directory in JsutDirectories:
        SourceFamily = "jsut_basic5000"
        Match = JsutIdPattern.search(FileName)
        SourceId = Match.group(1).upper() if Match else ""
    elif Directory == CommonVoiceDirectory:
        SourceFamily = "common_voice_prompt"
        # The nested generated/ tree is a byte-identical redundant copy of the
        # shorter root tree.  Both filenames are gen_<id>.wav; the datasheet also
        # says upstream reference data are not redistributed.  Keep the shorter
        # root path as the canonical generated member and exclude the nested copy.
        Role = "redundant_duplicate_copy" if len(Parts) >= 4 and Parts[2] == "generated" else "generated"
        Match = CommonVoiceIdPattern.search(FileName)
        SourceId = f"CVPROMPT-{int(Match.group(1)):05d}" if Match else ""
    GroupKey = f"{SourceFamily}:{SourceId}" if SourceId else f"unparsed:{Name}"
    return {
        "dataset_dir": Directory,
        "role": Role,
        "source_family": SourceFamily,
        "source_id": SourceId,
        "generator": Generator,
        "content_group_key": GroupKey,
        "recommended_content_split": StableSplit(GroupKey),
    }


def StableSplit(GroupKey: str) -> str:
    Digest = hashlib.sha256(f"{SplitSalt}|{GroupKey}".encode("utf-8")).digest()
    Value = int.from_bytes(Digest[:8], "big") / 2**64
    if Value < 0.8:
        return "train"
    if Value < 0.9:
        return "validation"
    return "test"


def ParseFmt(Data: bytes) -> dict[str, int | str]:
    if len(Data) < 16:
        raise ValueError("fmt chunk is shorter than 16 bytes")
    FormatTag, Channels, SampleRate, ByteRate, BlockAlign, Bits = struct.unpack_from("<HHIIHH", Data, 0)
    ValidBits = Bits
    EffectiveTag = FormatTag
    if FormatTag == 0xFFFE:
        if len(Data) < 40:
            raise ValueError("WAVE_FORMAT_EXTENSIBLE fmt chunk is shorter than 40 bytes")
        ValidBits = struct.unpack_from("<H", Data, 18)[0]
        EffectiveTag = struct.unpack_from("<I", Data, 24)[0]
    if EffectiveTag == 1:
        Codec = f"pcm_s{Bits}le" if Bits != 8 else "pcm_u8"
    elif EffectiveTag == 3:
        Codec = f"pcm_f{Bits}le"
    else:
        Codec = f"wave_format_{EffectiveTag}"
    return {
        "audio_format_tag": FormatTag,
        "effective_audio_format_tag": EffectiveTag,
        "codec": Codec,
        "channels": Channels,
        "sample_rate_hz": SampleRate,
        "byte_rate": ByteRate,
        "block_align": BlockAlign,
        "bits_per_sample": Bits,
        "valid_bits_per_sample": ValidBits,
    }


def DecodeSamples(Data: bytes, Format: dict[str, int | str]) -> np.ndarray:
    EffectiveTag = int(Format["effective_audio_format_tag"])
    Bits = int(Format["bits_per_sample"])
    if EffectiveTag == 1 and Bits == 8:
        return (np.frombuffer(Data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if EffectiveTag == 1 and Bits == 16:
        return np.frombuffer(Data, dtype="<i2").astype(np.float64) / 32768.0
    if EffectiveTag == 1 and Bits == 24:
        Packed = np.frombuffer(Data, dtype=np.uint8).reshape(-1, 3)
        Values = (
            Packed[:, 0].astype(np.int32)
            | (Packed[:, 1].astype(np.int32) << 8)
            | (Packed[:, 2].astype(np.int32) << 16)
        )
        Signed = np.where(Values & 0x800000, Values - 0x1000000, Values)
        return Signed.astype(np.float64) / 8_388_608.0
    if EffectiveTag == 1 and Bits == 32:
        return np.frombuffer(Data, dtype="<i4").astype(np.float64) / 2_147_483_648.0
    if EffectiveTag == 3 and Bits == 32:
        return np.frombuffer(Data, dtype="<f4").astype(np.float64)
    if EffectiveTag == 3 and Bits == 64:
        return np.frombuffer(Data, dtype="<f8")
    raise ValueError(f"Unsupported WAV sample representation: tag={EffectiveTag}, bits={Bits}")


def ConsumeDataChunk(
    Reader: TrackingReader,
    ChunkBytes: int,
    Format: dict[str, int | str],
    PcmDigest: hashlib._Hash,
) -> dict[str, int | float]:
    BlockAlign = int(Format["block_align"])
    if BlockAlign <= 0:
        raise ValueError("block_align must be positive")
    if ChunkBytes % BlockAlign:
        raise ValueError(f"data bytes {ChunkBytes} are not divisible by block_align {BlockAlign}")
    Remaining = ChunkBytes
    SampleCount = 0
    NonfiniteCount = 0
    SilentCount = 0
    ClippedCount = 0
    SumSquares = 0.0
    Peak = 0.0
    while Remaining:
        Take = min(Remaining, PcmBlockBytes)
        if Take < Remaining:
            Take -= Take % BlockAlign
        Data = Reader.ReadExact(Take)
        Remaining -= Take
        PcmDigest.update(Data)
        Samples = DecodeSamples(Data, Format)
        Finite = np.isfinite(Samples)
        NonfiniteCount += int(Samples.size - np.count_nonzero(Finite))
        FiniteSamples = Samples[Finite]
        Absolute = np.abs(FiniteSamples)
        SampleCount += int(Samples.size)
        if FiniteSamples.size:
            SumSquares += float(np.dot(FiniteSamples, FiniteSamples))
            SilentCount += int(np.count_nonzero(Absolute < SilenceThreshold))
            ClippedCount += int(np.count_nonzero(Absolute >= ClippingThreshold))
            Peak = max(Peak, float(Absolute.max()))
    return {
        "sample_count": SampleCount,
        "nonfinite_sample_count": NonfiniteCount,
        "silent_sample_count": SilentCount,
        "clipped_sample_count": ClippedCount,
        "sum_squares": SumSquares,
        "peak_abs_normalized": Peak,
    }


def MergeQuality(Total: dict[str, int | float], Part: dict[str, int | float]) -> None:
    for Name in (
        "sample_count",
        "nonfinite_sample_count",
        "silent_sample_count",
        "clipped_sample_count",
        "sum_squares",
    ):
        Total[Name] = Total.get(Name, 0) + Part.get(Name, 0)
    Total["peak_abs_normalized"] = max(
        float(Total.get("peak_abs_normalized", 0.0)),
        float(Part.get("peak_abs_normalized", 0.0)),
    )


def AuditEntry(Archive: zipfile.ZipFile, Info: zipfile.ZipInfo) -> dict[str, object]:
    Classification = ClassifyMember(Info.filename)
    Row: dict[str, object] = {
        "zip_member": Info.filename,
        **Classification,
        "compressed_bytes": Info.compress_size,
        "uncompressed_bytes": Info.file_size,
        "compression_method": Info.compress_type,
        "expected_crc32": f"{Info.CRC:08x}",
        "crc_status": "ERROR",
        "signature_status": "UNKNOWN",
        "parse_status": "ERROR",
        "parse_error": "",
        "license_id": "CC-BY-SA-4.0",
        "attribution_text": "WaveFake by Joel Frank and Lea Schönherr, Ruhr University Bochum, CC BY-SA 4.0",
        "changes_notice_required_when_shared": True,
        "sharealike_required_when_adapted": True,
        "training_eligible": False,
        "eligibility_reason": "PENDING_AUDIT",
        "issues": "",
    }
    Issues: list[str] = []
    Reader: TrackingReader | None = None
    PcmDigest = hashlib.sha256()
    Format: dict[str, int | str] | None = None
    Quality: dict[str, int | float] = {}
    DataBytes = 0
    FmtBytes = 0
    RiffDeclaredBytes: int | None = None
    try:
        with Archive.open(Info, "r") as Stream:
            Reader = TrackingReader(Stream)
            Header = Reader.ReadExact(12)
            if Header[:4] != b"RIFF" or Header[8:12] != b"WAVE":
                Row["signature_status"] = "NOT_RIFF_WAVE"
                raise ValueError(f"Expected RIFF/WAVE signature, got {Header[:4]!r}/{Header[8:12]!r}")
            Row["signature_status"] = "RIFF_WAVE"
            RiffDeclaredBytes = struct.unpack_from("<I", Header, 4)[0] + 8
            while True:
                ChunkHeader = Reader.Read(8)
                if not ChunkHeader:
                    break
                if len(ChunkHeader) != 8:
                    raise EOFError("Truncated WAV chunk header")
                ChunkId = ChunkHeader[:4]
                ChunkBytes = struct.unpack_from("<I", ChunkHeader, 4)[0]
                if ChunkId == b"fmt ":
                    FmtData = Reader.ReadExact(ChunkBytes)
                    FmtBytes += ChunkBytes
                    if Format is None:
                        Format = ParseFmt(FmtData)
                elif ChunkId == b"data":
                    if Format is None:
                        raise ValueError("data chunk appears before fmt chunk")
                    Part = ConsumeDataChunk(Reader, ChunkBytes, Format, PcmDigest)
                    MergeQuality(Quality, Part)
                    DataBytes += ChunkBytes
                else:
                    Remaining = ChunkBytes
                    while Remaining:
                        Part = Reader.ReadExact(min(Remaining, 8 * 1024 * 1024))
                        Remaining -= len(Part)
                if ChunkBytes % 2:
                    Reader.ReadExact(1)
            if Format is None:
                raise ValueError("Missing fmt chunk")
            if DataBytes == 0:
                raise ValueError("Missing or empty data chunk")
            Channels = int(Format["channels"])
            BlockAlign = int(Format["block_align"])
            Frames = DataBytes // BlockAlign
            ExpectedSamples = Frames * Channels
            if int(Quality.get("sample_count", 0)) != ExpectedSamples:
                raise ValueError(
                    f"Decoded samples {Quality.get('sample_count', 0)} != expected {ExpectedSamples}"
                )
            Row["parse_status"] = "OK"
    except Exception as Error:
        Row["parse_error"] = f"{type(Error).__name__}: {Error}"
        Issues.append("parse_error")
        if Reader is not None:
            try:
                Reader.Drain()
            except Exception as DrainError:
                Issues.append(f"drain_error={type(DrainError).__name__}")
    if Reader is not None:
        Row["bytes_read"] = Reader.BytesRead
        Row["actual_crc32"] = f"{Reader.Crc32 & 0xFFFFFFFF:08x}"
        Row["file_sha256"] = Reader.FileDigest.hexdigest()
        Row["crc_status"] = "OK" if (Reader.Crc32 & 0xFFFFFFFF) == Info.CRC else "MISMATCH"
        if Reader.BytesRead != Info.file_size:
            Issues.append("uncompressed_size_mismatch")
        if Row["crc_status"] != "OK":
            Issues.append("crc_mismatch")
    Row["riff_declared_bytes"] = RiffDeclaredBytes if RiffDeclaredBytes is not None else ""
    Row["fmt_chunk_bytes"] = FmtBytes
    Row["data_bytes"] = DataBytes
    if RiffDeclaredBytes is not None and RiffDeclaredBytes != Info.file_size:
        Issues.append("riff_declared_size_mismatch")
    if Format is not None:
        for Name in (
            "audio_format_tag",
            "codec",
            "channels",
            "sample_rate_hz",
            "bits_per_sample",
            "valid_bits_per_sample",
            "block_align",
            "byte_rate",
        ):
            Row[Name] = Format.get(Name, "")
        BlockAlign = int(Format["block_align"])
        SampleRate = int(Format["sample_rate_hz"])
        Frames = DataBytes // BlockAlign if BlockAlign else 0
        Row["frames"] = Frames
        Row["duration_seconds"] = Frames / SampleRate if SampleRate else ""
    else:
        Row["frames"] = ""
        Row["duration_seconds"] = ""
    SampleCount = int(Quality.get("sample_count", 0))
    FiniteCount = SampleCount - int(Quality.get("nonfinite_sample_count", 0))
    Row["pcm_sha256"] = PcmDigest.hexdigest() if DataBytes else ""
    Row["sample_count"] = SampleCount
    Row["nonfinite_sample_count"] = int(Quality.get("nonfinite_sample_count", 0))
    Row["peak_abs_normalized"] = float(Quality.get("peak_abs_normalized", 0.0))
    Row["rms_normalized"] = (
        math.sqrt(float(Quality.get("sum_squares", 0.0)) / FiniteCount) if FiniteCount else ""
    )
    Row["silent_sample_count"] = int(Quality.get("silent_sample_count", 0))
    Row["silent_sample_fraction_lt_1e_4"] = (
        int(Quality.get("silent_sample_count", 0)) / FiniteCount if FiniteCount else ""
    )
    Row["clipped_sample_count"] = int(Quality.get("clipped_sample_count", 0))
    Row["clipped_sample_fraction_ge_0_999"] = (
        int(Quality.get("clipped_sample_count", 0)) / FiniteCount if FiniteCount else ""
    )
    if not Row["source_id"]:
        Issues.append("source_id_unparsed")
    if Row["role"] == "redundant_duplicate_copy":
        Row["training_eligible"] = False
        Row["eligibility_reason"] = "EXCLUDE_REDUNDANT_EXACT_DUPLICATE_COPY"
    elif Row["parse_status"] == "OK" and Row["crc_status"] == "OK":
        Row["training_eligible"] = True
        Row["eligibility_reason"] = "GO_WAVEFAKE_CC_BY_SA_4_WITH_ATTRIBUTION_SA_AND_CHANGES_NOTICE"
    else:
        Row["training_eligible"] = False
        Row["eligibility_reason"] = "EXCLUDE_TECHNICAL_AUDIT_FAILURE"
    Row["issues"] = ";".join(Issues)
    return Row


def WriteProgress(PathValue: Path, Payload: dict[str, object]) -> None:
    Temporary = PathValue.with_suffix(PathValue.suffix + ".tmp")
    Temporary.write_text(json.dumps(Payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(Temporary, PathValue)


def LoadRows(PathValue: Path) -> list[dict[str, str]]:
    if not PathValue.exists():
        return []
    with PathValue.open("r", encoding="utf-8-sig", newline="") as Stream:
        return list(csv.DictReader(Stream))


def WriteCsv(PathValue: Path, Rows: list[dict[str, object]], Fields: list[str]) -> None:
    with PathValue.open("w", encoding="utf-8-sig", newline="") as Stream:
        Writer = csv.DictWriter(Stream, fieldnames=Fields, extrasaction="ignore")
        Writer.writeheader()
        Writer.writerows(Rows)


def Quantiles(Values: list[float]) -> dict[str, float | int | None]:
    if not Values:
        return {"count": 0, "min": None, "p50": None, "max": None}
    Array = np.asarray(Values, dtype=np.float64)
    return {
        "count": int(Array.size),
        "min": float(Array.min()),
        "p01": float(np.quantile(Array, 0.01)),
        "p05": float(np.quantile(Array, 0.05)),
        "p25": float(np.quantile(Array, 0.25)),
        "p50": float(np.quantile(Array, 0.50)),
        "p75": float(np.quantile(Array, 0.75)),
        "p95": float(np.quantile(Array, 0.95)),
        "p99": float(np.quantile(Array, 0.99)),
        "max": float(Array.max()),
        "mean": float(Array.mean()),
        "std_population": float(Array.std()),
    }


def IntValue(Row: dict[str, str], Name: str) -> int:
    Value = Row.get(Name, "")
    return int(Value) if Value not in ("", None) else 0


def FloatValue(Row: dict[str, str], Name: str) -> float:
    Value = Row.get(Name, "")
    return float(Value) if Value not in ("", None) else 0.0


def BoolValue(Row: dict[str, str], Name: str) -> bool:
    return str(Row.get(Name, "")).casefold() == "true"


def NormalizeCommonVoiceRows(Rows: list[dict[str, str]]) -> None:
    """Apply the canonical/duplicate role rule to resumed inventories."""
    for Row in Rows:
        if Row.get("dataset_dir") != CommonVoiceDirectory:
            continue
        Parts = Row["zip_member"].replace("\\", "/").split("/")
        IsNestedCopy = len(Parts) >= 4 and Parts[2] == "generated"
        Row["role"] = "redundant_duplicate_copy" if IsNestedCopy else "generated"
        Row["generator"] = GeneratorByDirectory[CommonVoiceDirectory]
        if IsNestedCopy:
            Row["training_eligible"] = False
            Row["eligibility_reason"] = "EXCLUDE_REDUNDANT_EXACT_DUPLICATE_COPY"
        elif Row["parse_status"] == "OK" and Row["crc_status"] == "OK":
            Row["training_eligible"] = True
            Row["eligibility_reason"] = "GO_WAVEFAKE_CC_BY_SA_4_WITH_ATTRIBUTION_SA_AND_CHANGES_NOTICE"
        else:
            Row["training_eligible"] = False
            Row["eligibility_reason"] = "EXCLUDE_TECHNICAL_AUDIT_FAILURE"


def BuildSourceGroups(Rows: list[dict[str, str]]) -> list[dict[str, object]]:
    Groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for Row in Rows:
        Groups[Row["content_group_key"]].append(Row)
    Output: list[dict[str, object]] = []
    for GroupKey, Members in sorted(Groups.items()):
        Family = Members[0]["source_family"]
        SourceId = Members[0]["source_id"]
        Generated = [Row for Row in Members if Row["role"] == "generated"]
        DuplicateCopies = [Row for Row in Members if Row["role"] == "redundant_duplicate_copy"]
        ExpectedGenerated = 7 if Family == "ljspeech" else 2 if Family == "jsut_basic5000" else 1
        Output.append(
            {
                "content_group_key": GroupKey,
                "source_family": Family,
                "source_id": SourceId,
                "recommended_content_split": Members[0]["recommended_content_split"],
                "generated_member_count": len(Generated),
                "redundant_duplicate_copy_count": len(DuplicateCopies),
                "expected_generated_member_count": ExpectedGenerated,
                "complete_expected_generator_coverage": len(Generated) == ExpectedGenerated,
                "all_generated_technical_pass": all(
                    Row["parse_status"] == "OK" and Row["crc_status"] == "OK" for Row in Generated
                ),
                "generators": ";".join(sorted({Row["generator"] for Row in Generated})),
                "zip_members": ";".join(sorted(Row["zip_member"] for Row in Members)),
            }
        )
    return Output


def BuildLjsPairing(
    Rows: list[dict[str, str]],
    LjsInventoryPath: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    LjsRows = LoadRows(LjsInventoryPath)
    LjsById = {Row["id"]: Row for Row in LjsRows if Row.get("parse_status") == "OK"}
    GeneratedById: defaultdict[str, dict[str, str]] = defaultdict(dict)
    DuplicateKeys: list[str] = []
    for Row in Rows:
        if Row["source_family"] != "ljspeech" or Row["role"] != "generated":
            continue
        SourceId = Row["source_id"]
        Directory = Row["dataset_dir"]
        if Directory in GeneratedById[SourceId]:
            DuplicateKeys.append(f"{Directory}:{SourceId}")
        GeneratedById[SourceId][Directory] = Row["zip_member"]
    Output: list[dict[str, object]] = []
    for SourceId in sorted(LjsById):
        Generated = GeneratedById.get(SourceId, {})
        Row: dict[str, object] = {
            "ljspeech_id": SourceId,
            "content_group_key": f"ljspeech:{SourceId}",
            "recommended_content_split": StableSplit(f"ljspeech:{SourceId}"),
            "real_relative_path": LjsById[SourceId]["relative_path"],
            "real_file_sha256": LjsById[SourceId]["file_sha256"],
            "generated_count": len(Generated),
            "complete_7_generator_pair": len(Generated) == len(LjsDirectories),
        }
        for Directory in LjsDirectories:
            Row[Directory] = Generated.get(Directory, "")
        Output.append(Row)
    GeneratedIds = set(GeneratedById)
    RealIds = set(LjsById)
    Summary = {
        "ljspeech_real_id_count": len(RealIds),
        "wavefake_ljspeech_source_id_count": len(GeneratedIds),
        "missing_generated_source_id_count": len(RealIds - GeneratedIds),
        "extra_generated_source_id_count": len(GeneratedIds - RealIds),
        "complete_7_generator_pair_count": sum(bool(Row["complete_7_generator_pair"]) for Row in Output),
        "duplicate_directory_source_id_count": len(DuplicateKeys),
        "missing_generated_source_ids": sorted(RealIds - GeneratedIds),
        "extra_generated_source_ids": sorted(GeneratedIds - RealIds),
        "duplicate_directory_source_ids": sorted(DuplicateKeys),
    }
    return Output, Summary


def BuildPairingSummary(
    Rows: list[dict[str, str]],
    LjsSummary: dict[str, object],
) -> list[dict[str, object]]:
    Output: list[dict[str, object]] = []
    for Directory in LjsDirectories + JsutDirectories:
        Members = [Row for Row in Rows if Row["dataset_dir"] == Directory]
        SourceIds = [Row["source_id"] for Row in Members]
        Expected = 13_100 if Directory in LjsDirectories else 5_000
        Output.append(
            {
                "dataset_dir": Directory,
                "role": "generated",
                "source_family": Members[0]["source_family"] if Members else "",
                "member_count": len(Members),
                "unique_source_id_count": len(set(SourceIds)),
                "expected_source_id_count": Expected,
                "duplicate_source_id_count": len(SourceIds) - len(set(SourceIds)),
                "missing_against_family_union_count": "",
                "pairing_status": "PASS" if len(Members) == Expected and len(set(SourceIds)) == Expected else "FAIL",
            }
        )
    for Role in ("generated", "redundant_duplicate_copy"):
        Members = [
            Row
            for Row in Rows
            if Row["dataset_dir"] == CommonVoiceDirectory and Row["role"] == Role
        ]
        OtherRole = "redundant_duplicate_copy" if Role == "generated" else "generated"
        OtherIds = {
            Row["source_id"]
            for Row in Rows
            if Row["dataset_dir"] == CommonVoiceDirectory and Row["role"] == OtherRole
        }
        SourceIds = [Row["source_id"] for Row in Members]
        Output.append(
            {
                "dataset_dir": CommonVoiceDirectory,
                "role": Role,
                "source_family": "common_voice_prompt",
                "member_count": len(Members),
                "unique_source_id_count": len(set(SourceIds)),
                "expected_source_id_count": 16_283,
                "duplicate_source_id_count": len(SourceIds) - len(set(SourceIds)),
                "missing_against_family_union_count": len(set(SourceIds) - OtherIds),
                "pairing_status": "PASS"
                if len(Members) == 16_283 and len(set(SourceIds)) == 16_283 and set(SourceIds) == OtherIds
                else "FAIL",
            }
        )
    Output.append(
        {
            "dataset_dir": "LJSPEECH_REAL_JOIN",
            "role": "real_plus_7_generated",
            "source_family": "ljspeech",
            "member_count": int(LjsSummary["ljspeech_real_id_count"]) * 8,
            "unique_source_id_count": LjsSummary["ljspeech_real_id_count"],
            "expected_source_id_count": 13_100,
            "duplicate_source_id_count": LjsSummary["duplicate_directory_source_id_count"],
            "missing_against_family_union_count": LjsSummary["missing_generated_source_id_count"],
            "pairing_status": "PASS"
            if int(LjsSummary["complete_7_generator_pair_count"]) == 13_100
            else "FAIL",
        }
    )
    return Output


def Main() -> int:
    Args = ParseArgs()
    ArchivePath = Args.archive.resolve()
    OutputDir = Args.output_dir.resolve()
    OutputDir.mkdir(parents=True, exist_ok=True)
    InventoryPath = OutputDir / "wavefake-audio-inventory.csv"
    ProgressPath = OutputDir / "wavefake-audit-progress.json"
    SourceStateBefore = FileState(ArchivePath)
    with Args.download_run.resolve().open("r", encoding="utf-8") as Stream:
        DownloadRun = json.load(Stream)
    if SourceStateBefore["size_bytes"] != ExpectedArchiveBytes:
        raise RuntimeError("WaveFake archive size does not match the verified official archive")
    if DownloadRun.get("md5", "").casefold() != ExpectedArchiveMd5:
        raise RuntimeError("Verified download record MD5 does not match official MD5")
    if DownloadRun.get("sha256", "").casefold() != ExpectedArchiveSha256:
        raise RuntimeError("Verified download record SHA-256 does not match expected SHA-256")

    ExistingRows = LoadRows(InventoryPath) if Args.resume else []
    if InventoryPath.exists() and not Args.resume:
        raise FileExistsError("Inventory already exists; pass --resume to continue without overwriting")
    Processed = {Row["zip_member"] for Row in ExistingRows}
    StartTime = time.time()
    with zipfile.ZipFile(ArchivePath, "r") as Archive:
        Infos = [Info for Info in Archive.infolist() if not Info.is_dir()]
        Central = {
            "file_member_count": len(Infos),
            "compressed_bytes": sum(Info.compress_size for Info in Infos),
            "uncompressed_bytes": sum(Info.file_size for Info in Infos),
            "unsafe_path_count": sum(IsUnsafePath(Info.filename) for Info in Infos),
            "symlink_count": sum(IsSymlink(Info) for Info in Infos),
            "compression_method_counts": dict(Counter(str(Info.compress_type) for Info in Infos)),
            "directory_counts": dict(
                sorted(Counter(ClassifyMember(Info.filename)["dataset_dir"] for Info in Infos).items())
            ),
        }
        if len(Infos) != ExpectedMemberCount:
            raise RuntimeError("Unexpected WaveFake ZIP member count")
        if Central["compressed_bytes"] != ExpectedCompressedBytes:
            raise RuntimeError("Unexpected WaveFake compressed member bytes")
        if Central["uncompressed_bytes"] != ExpectedUncompressedBytes:
            raise RuntimeError("Unexpected WaveFake uncompressed member bytes")
        if Central["unsafe_path_count"] or Central["symlink_count"]:
            raise RuntimeError("Unsafe ZIP member or symlink found")
        InfoNames = {Info.filename for Info in Infos}
        if not Processed.issubset(InfoNames):
            raise RuntimeError("Resume inventory contains members not found in the current archive")
        Mode = "a" if ExistingRows else "w"
        Encoding = "utf-8" if ExistingRows else "utf-8-sig"
        with InventoryPath.open(Mode, encoding=Encoding, newline="") as InventoryStream:
            Writer = csv.DictWriter(
                InventoryStream,
                fieldnames=InventoryFields,
                extrasaction="ignore",
            )
            if not ExistingRows:
                Writer.writeheader()
            Completed = len(Processed)
            for Info in Infos:
                if Info.filename in Processed:
                    continue
                Row = AuditEntry(Archive, Info)
                Writer.writerow(Row)
                Completed += 1
                if Completed % ProgressInterval == 0 or Completed == len(Infos):
                    InventoryStream.flush()
                    WriteProgress(
                        ProgressPath,
                        {
                            "status": "RUNNING" if Completed < len(Infos) else "CONTENT_STREAM_COMPLETE",
                            "completed_members": Completed,
                            "total_members": len(Infos),
                            "percent": Completed / len(Infos) * 100,
                            "elapsed_seconds_this_run": time.time() - StartTime,
                            "archive_state": SourceStateBefore,
                        },
                    )
                    print(f"audited {Completed}/{len(Infos)} WAV members", flush=True)

    Rows = LoadRows(InventoryPath)
    if len(Rows) != ExpectedMemberCount:
        raise RuntimeError(f"Inventory row count {len(Rows)} != {ExpectedMemberCount}")
    if len({Row["zip_member"] for Row in Rows}) != ExpectedMemberCount:
        raise RuntimeError("Inventory ZIP member keys are not unique")
    NormalizeCommonVoiceRows(Rows)
    # Persist corrected roles/eligibility for both fresh and resumed runs.
    WriteCsv(InventoryPath, Rows, InventoryFields)

    OkRows = [Row for Row in Rows if Row["parse_status"] == "OK" and Row["crc_status"] == "OK"]
    GeneratedRows = [Row for Row in Rows if Row["role"] == "generated"]
    DuplicateCopyRows = [Row for Row in Rows if Row["role"] == "redundant_duplicate_copy"]
    EligibleRows = [Row for Row in Rows if BoolValue(Row, "training_eligible")]
    FileGroups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    PcmGroups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for Row in OkRows:
        FileGroups[Row["file_sha256"]].append(Row)
        PcmGroups[Row["pcm_sha256"]].append(Row)
    DuplicateFileGroups = {Key: Value for Key, Value in FileGroups.items() if len(Value) > 1}
    DuplicatePcmGroups = {Key: Value for Key, Value in PcmGroups.items() if len(Value) > 1}

    DuplicateRows: list[dict[str, object]] = []
    for DuplicateType, Groups in (
        ("file_sha256", DuplicateFileGroups),
        ("pcm_sha256", DuplicatePcmGroups),
    ):
        for Digest, Members in sorted(Groups.items()):
            DuplicateRows.append(
                {
                    "duplicate_type": DuplicateType,
                    "sha256": Digest,
                    "member_count": len(Members),
                    "source_families": ";".join(sorted({Row["source_family"] for Row in Members})),
                    "roles": ";".join(sorted({Row["role"] for Row in Members})),
                    "dataset_dirs": ";".join(sorted({Row["dataset_dir"] for Row in Members})),
                    "source_ids": ";".join(sorted({Row["source_id"] for Row in Members})),
                    "zip_members": ";".join(sorted(Row["zip_member"] for Row in Members)),
                }
            )
    WriteCsv(
        OutputDir / "wavefake-duplicate-groups.csv",
        DuplicateRows,
        [
            "duplicate_type",
            "sha256",
            "member_count",
            "source_families",
            "roles",
            "dataset_dirs",
            "source_ids",
            "zip_members",
        ],
    )

    SourceGroups = BuildSourceGroups(Rows)
    WriteCsv(
        OutputDir / "wavefake-source-group-manifest.csv",
        SourceGroups,
        list(SourceGroups[0]),
    )
    LjsPairing, LjsSummary = BuildLjsPairing(Rows, Args.ljspeech_inventory.resolve())
    LjsPairFields = [
        "ljspeech_id",
        "content_group_key",
        "recommended_content_split",
        "real_relative_path",
        "real_file_sha256",
        "generated_count",
        "complete_7_generator_pair",
        *LjsDirectories,
    ]
    WriteCsv(OutputDir / "wavefake-ljspeech-pairing.csv", LjsPairing, LjsPairFields)
    PairingRows = BuildPairingSummary(Rows, LjsSummary)
    WriteCsv(OutputDir / "wavefake-pairing-summary.csv", PairingRows, list(PairingRows[0]))

    GeneratorSummaryRows: list[dict[str, object]] = []
    for (Directory, Role), Members in sorted(
        defaultdict(list, {
            Key: [Row for Row in Rows if (Row["dataset_dir"], Row["role"]) == Key]
            for Key in sorted({(Row["dataset_dir"], Row["role"]) for Row in Rows})
        }).items()
    ):
        SampleCount = sum(IntValue(Row, "sample_count") for Row in Members)
        SilentCount = sum(IntValue(Row, "silent_sample_count") for Row in Members)
        ClippedCount = sum(IntValue(Row, "clipped_sample_count") for Row in Members)
        Durations = [FloatValue(Row, "duration_seconds") for Row in Members if Row["duration_seconds"]]
        GeneratorSummaryRows.append(
            {
                "dataset_dir": Directory,
                "role": Role,
                "source_family": Members[0]["source_family"],
                "generator": Members[0]["generator"],
                "member_count": len(Members),
                "unique_source_id_count": len({Row["source_id"] for Row in Members}),
                "crc_ok_count": sum(Row["crc_status"] == "OK" for Row in Members),
                "parse_ok_count": sum(Row["parse_status"] == "OK" for Row in Members),
                "training_eligible_count": sum(BoolValue(Row, "training_eligible") for Row in Members),
                "compressed_bytes": sum(IntValue(Row, "compressed_bytes") for Row in Members),
                "uncompressed_bytes": sum(IntValue(Row, "uncompressed_bytes") for Row in Members),
                "codec_counts": json.dumps(dict(Counter(Row["codec"] for Row in Members)), sort_keys=True),
                "sample_rate_counts": json.dumps(
                    dict(Counter(Row["sample_rate_hz"] for Row in Members)), sort_keys=True
                ),
                "channel_counts": json.dumps(dict(Counter(Row["channels"] for Row in Members)), sort_keys=True),
                "bits_per_sample_counts": json.dumps(
                    dict(Counter(Row["bits_per_sample"] for Row in Members)), sort_keys=True
                ),
                "total_duration_seconds": sum(Durations),
                "duration_min_seconds": min(Durations) if Durations else "",
                "duration_p50_seconds": float(np.quantile(Durations, 0.5)) if Durations else "",
                "duration_max_seconds": max(Durations) if Durations else "",
                "weighted_silence_fraction_lt_1e_4": SilentCount / SampleCount if SampleCount else "",
                "silence_ge_50pct_count": sum(
                    FloatValue(Row, "silent_sample_fraction_lt_1e_4") >= 0.5 for Row in Members
                ),
                "weighted_clipping_fraction_ge_0_999": ClippedCount / SampleCount if SampleCount else "",
                "clipping_ge_1pct_count": sum(
                    FloatValue(Row, "clipped_sample_fraction_ge_0_999") >= 0.01 for Row in Members
                ),
            }
        )
    WriteCsv(
        OutputDir / "wavefake-generator-summary.csv",
        GeneratorSummaryRows,
        list(GeneratorSummaryRows[0]),
    )

    SourceStateAfter = FileState(ArchivePath)
    TotalSamples = sum(IntValue(Row, "sample_count") for Row in OkRows)
    TotalSilent = sum(IntValue(Row, "silent_sample_count") for Row in OkRows)
    TotalClipped = sum(IntValue(Row, "clipped_sample_count") for Row in OkRows)
    DurationValues = [FloatValue(Row, "duration_seconds") for Row in OkRows]
    SilenceValues = [FloatValue(Row, "silent_sample_fraction_lt_1e_4") for Row in OkRows]
    ClippingValues = [FloatValue(Row, "clipped_sample_fraction_ge_0_999") for Row in OkRows]
    EligibleSamples = sum(IntValue(Row, "sample_count") for Row in EligibleRows)
    EligibleDurations = [FloatValue(Row, "duration_seconds") for Row in EligibleRows]
    EligibleSilent = sum(IntValue(Row, "silent_sample_count") for Row in EligibleRows)
    EligibleClipped = sum(IntValue(Row, "clipped_sample_count") for Row in EligibleRows)
    CommonDuplicateIds = {
        Row["source_id"] for Row in DuplicateCopyRows if Row["source_family"] == "common_voice_prompt"
    }
    CommonGeneratedIds = {
        Row["source_id"]
        for Row in GeneratedRows
        if Row["source_family"] == "common_voice_prompt"
    }
    JsutGroups = [Row for Row in SourceGroups if Row["source_family"] == "jsut_basic5000"]
    AllGeneratedTechnicalPass = all(
        Row["parse_status"] == "OK" and Row["crc_status"] == "OK" for Row in GeneratedRows
    )
    CommonCanonicalById = {
        Row["source_id"]: Row
        for Row in GeneratedRows
        if Row["source_family"] == "common_voice_prompt"
    }
    CommonDuplicateById = {
        Row["source_id"]: Row
        for Row in DuplicateCopyRows
        if Row["source_family"] == "common_voice_prompt"
    }
    CommonCopiesExact = (
        set(CommonCanonicalById) == set(CommonDuplicateById)
        and len(CommonCanonicalById) == 16_283
        and all(
            CommonCanonicalById[SourceId]["file_sha256"] == CommonDuplicateById[SourceId]["file_sha256"]
            and CommonCanonicalById[SourceId]["pcm_sha256"] == CommonDuplicateById[SourceId]["pcm_sha256"]
            for SourceId in CommonCanonicalById
        )
    )
    PairingPass = (
        int(LjsSummary["complete_7_generator_pair_count"]) == 13_100
        and all(bool(Row["complete_expected_generator_coverage"]) for Row in JsutGroups)
        and CommonDuplicateIds == CommonGeneratedIds
        and len(CommonGeneratedIds) == 16_283
        and CommonCopiesExact
    )
    Readiness = (
        "READY"
        if len(Rows) == ExpectedMemberCount
        and len(GeneratedRows) == 117_983
        and len(DuplicateCopyRows) == 16_283
        and len(EligibleRows) == 117_983
        and AllGeneratedTechnicalPass
        and PairingPass
        and SourceStateBefore == SourceStateAfter
        else "BLOCKED"
    )
    Audit = {
        "data_readiness": Readiness,
        "block_reason": "" if Readiness == "READY" else "WaveFake full streaming integrity, audio, or pairing gates did not all pass.",
        "dataset": "WaveFake 1.2.0 official generated_audio.zip",
        "method": {
            "scope": "all 134266 ZIP file members; no sampling and no persistent extraction",
            "streaming": "each ZIP member read to EOF once in central-directory order",
            "crc": "Python ZipExtFile verification plus independently accumulated CRC-32",
            "file_duplicate_key": "SHA-256 of complete uncompressed WAV member bytes",
            "pcm_duplicate_key": "SHA-256 of concatenated WAV data-chunk bytes",
            "silence_definition": "abs(normalized sample) < 1e-4",
            "clipping_definition": "abs(normalized sample) >= 0.999",
            "content_split": "SHA-256 deterministic 80/10/10 by source-family source-ID group",
            "content_split_salt": SplitSalt,
        },
        "source": {
            "archive": str(ArchivePath),
            "archive_bytes": ArchivePath.stat().st_size,
            "official_md5": ExpectedArchiveMd5,
            "verified_local_md5": DownloadRun.get("md5"),
            "md5_matches": DownloadRun.get("md5", "").casefold() == ExpectedArchiveMd5,
            "local_sha256": DownloadRun.get("sha256"),
            "expected_sha256": ExpectedArchiveSha256,
            "sha256_matches": DownloadRun.get("sha256", "").casefold() == ExpectedArchiveSha256,
            "central_directory": Central,
            "state_before": SourceStateBefore,
            "state_after": SourceStateAfter,
            "source_state_unchanged": SourceStateBefore == SourceStateAfter,
        },
        "composition": {
            "total_audio_members": len(Rows),
            "generated_audio_members": len(GeneratedRows),
            "redundant_duplicate_copy_members": len(DuplicateCopyRows),
            "source_reference_members": 0,
            "training_eligible_generated_members": len(EligibleRows),
            "directory_count": len({Row["dataset_dir"] for Row in Rows}),
            "source_family_counts": dict(Counter(Row["source_family"] for Row in Rows)),
            "role_counts": dict(Counter(Row["role"] for Row in Rows)),
            "datasheet_reported_generated_count": 117_985,
            "paper_reported_count": 104_885,
            "measured_unique_generated_count": len(GeneratedRows),
            "measured_total_zip_audio_count": len(Rows),
            "count_reconciliation": {
                "104885_to_117985": "13100 HiFi-GAN LJSpeech clips were added relative to the paper-era count",
                "117985_to_117983": "datasheet says 16285 TTS phrases while the archive contains 16283 unique generated TTS IDs (two fewer)",
                "117983_to_134266": "archive stores the same 16283 Common Voice-prompt generated WAVs twice; the nested generated/ paths are byte- and PCM-identical redundant copies",
            },
        },
        "integrity": {
            "inventory_rows": len(Rows),
            "unique_zip_members": len({Row["zip_member"] for Row in Rows}),
            "crc_ok_count": sum(Row["crc_status"] == "OK" for Row in Rows),
            "crc_failure_count": sum(Row["crc_status"] != "OK" for Row in Rows),
            "parse_ok_count": sum(Row["parse_status"] == "OK" for Row in Rows),
            "parse_failure_count": sum(Row["parse_status"] != "OK" for Row in Rows),
            "signature_counts": dict(Counter(Row["signature_status"] for Row in Rows)),
            "uncompressed_size_mismatch_count": sum(
                IntValue(Row, "bytes_read") != IntValue(Row, "uncompressed_bytes") for Row in Rows
            ),
            "riff_declared_size_mismatch_count": sum(
                "riff_declared_size_mismatch" in Row["issues"] for Row in Rows
            ),
        },
        "audio": {
            "codec_counts": dict(Counter(Row["codec"] for Row in OkRows)),
            "sample_rate_counts": dict(Counter(Row["sample_rate_hz"] for Row in OkRows)),
            "channel_counts": dict(Counter(Row["channels"] for Row in OkRows)),
            "bits_per_sample_counts": dict(Counter(Row["bits_per_sample"] for Row in OkRows)),
            "total_duration_seconds": sum(DurationValues),
            "total_duration_hours": sum(DurationValues) / 3600,
            "duration_seconds_distribution": Quantiles(DurationValues),
            "total_sample_count": TotalSamples,
            "nonfinite_sample_count": sum(IntValue(Row, "nonfinite_sample_count") for Row in OkRows),
            "weighted_silence_fraction_lt_1e_4": TotalSilent / TotalSamples if TotalSamples else None,
            "silence_fraction_distribution": Quantiles(SilenceValues),
            "silence_ge_50pct_count": sum(Value >= 0.5 for Value in SilenceValues),
            "weighted_clipping_fraction_ge_0_999": TotalClipped / TotalSamples if TotalSamples else None,
            "clipping_fraction_distribution": Quantiles(ClippingValues),
            "clipping_ge_1pct_count": sum(Value >= 0.01 for Value in ClippingValues),
            "file_sha256_duplicate_group_count": len(DuplicateFileGroups),
            "file_sha256_duplicate_affected_count": sum(len(Value) for Value in DuplicateFileGroups.values()),
            "pcm_sha256_duplicate_group_count": len(DuplicatePcmGroups),
            "pcm_sha256_duplicate_affected_count": sum(len(Value) for Value in DuplicatePcmGroups.values()),
            "training_eligible_unique_generated": {
                "member_count": len(EligibleRows),
                "total_duration_seconds": sum(EligibleDurations),
                "total_duration_hours": sum(EligibleDurations) / 3600,
                "duration_seconds_distribution": Quantiles(EligibleDurations),
                "total_sample_count": EligibleSamples,
                "nonfinite_sample_count": sum(
                    IntValue(Row, "nonfinite_sample_count") for Row in EligibleRows
                ),
                "weighted_silence_fraction_lt_1e_4": (
                    EligibleSilent / EligibleSamples if EligibleSamples else None
                ),
                "silence_ge_50pct_count": sum(
                    FloatValue(Row, "silent_sample_fraction_lt_1e_4") >= 0.5
                    for Row in EligibleRows
                ),
                "weighted_clipping_fraction_ge_0_999": (
                    EligibleClipped / EligibleSamples if EligibleSamples else None
                ),
                "clipping_ge_1pct_count": sum(
                    FloatValue(Row, "clipped_sample_fraction_ge_0_999") >= 0.01
                    for Row in EligibleRows
                ),
            },
        },
        "pairing": {
            **LjsSummary,
            "jsut_source_group_count": len(JsutGroups),
            "jsut_complete_2_generator_group_count": sum(
                bool(Row["complete_expected_generator_coverage"]) for Row in JsutGroups
            ),
            "common_voice_canonical_generated_id_count": len(CommonGeneratedIds),
            "common_voice_redundant_duplicate_id_count": len(CommonDuplicateIds),
            "common_voice_duplicate_paths_exact_file_and_pcm_match": CommonCopiesExact,
            "common_voice_upstream_reference_member_count": 0,
        },
        "license": {
            "dataset_license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "sharealike_required_for_adapted_material": True,
            "changes_notice_required": True,
            "generated_training_status": "GO_WITH_ATTRIBUTION_SA_AND_CHANGES_NOTICE_AFTER_EXACT_DEDUPLICATION",
            "common_voice_duplicate_copy_status": "EXCLUDED_REDUNDANT_COPY; canonical shorter root path retained",
            "common_voice_upstream_reference_status": "NOT_PRESENT, consistent with the datasheet statement that reference data are not redistributed",
        },
    }
    (OutputDir / "wavefake-audit-run.json").write_text(
        json.dumps(Audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    WriteProgress(
        ProgressPath,
        {
            "status": "COMPLETE",
            "completed_members": len(Rows),
            "total_members": ExpectedMemberCount,
            "percent": 100.0,
            "data_readiness": Readiness,
            "source_state_unchanged": SourceStateBefore == SourceStateAfter,
        },
    )
    print(
        json.dumps(
            {
                "data_readiness": Readiness,
                "inventory_rows": len(Rows),
                "generated": len(GeneratedRows),
                "redundant_duplicate_copy": len(DuplicateCopyRows),
                "crc_failures": Audit["integrity"]["crc_failure_count"],
                "parse_failures": Audit["integrity"]["parse_failure_count"],
                "duration_hours": Audit["audio"]["total_duration_hours"],
                "file_duplicate_groups": len(DuplicateFileGroups),
                "pcm_duplicate_groups": len(DuplicatePcmGroups),
                "source_unchanged": SourceStateBefore == SourceStateAfter,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
