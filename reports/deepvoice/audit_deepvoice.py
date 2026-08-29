"""<summary>Reproducible integrity and metadata audit for DeepVoice raw data</summary>"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


OfficialDataUrl = "https://dacon.io/competitions/official/236749/data"
ExpectedProbabilityColumns = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]
ExpectedSubmissionColumns = ["ID", *ExpectedProbabilityColumns]
AudioSuffixes = {".wav", ".flac", ".mp3"}


def ParseArguments() -> argparse.Namespace:
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--raw-dir", dest="RawDirectory", type=Path, required=True)
    Parser.add_argument("--output-dir", dest="OutputDirectory", type=Path, required=True)
    return Parser.parse_args()


def CalculateSha256(FilePath: Path) -> str:
    Digest = hashlib.sha256()
    with FilePath.open("rb") as FileHandle:
        while True:
            Block = FileHandle.read(1024 * 1024)
            if not Block:
                break
            Digest.update(Block)
    return Digest.hexdigest()


def DetectSignature(FilePath: Path) -> str:
    with FilePath.open("rb") as FileHandle:
        Header = FileHandle.read(16)
    if Header.startswith(b"RIFF") and Header[8:12] == b"WAVE":
        return "WAV"
    if Header.startswith(b"fLaC"):
        return "FLAC"
    if Header.startswith(b"ID3"):
        return "MP3_ID3"
    if len(Header) >= 2 and Header[0] == 0xFF and Header[1] & 0xE0 == 0xE0:
        return "MP3_FRAME"
    return "UNKNOWN"


def IsSafeArchiveMember(MemberName: str) -> bool:
    Normalized = MemberName.replace("\\", "/")
    Candidate = PurePosixPath(Normalized)
    if Candidate.is_absolute():
        return False
    if ".." in Candidate.parts:
        return False
    if Candidate.parts and ":" in Candidate.parts[0]:
        return False
    return True


def AuditZip(FilePath: Path) -> dict[str, Any]:
    Result: dict[str, Any] = {
        "Status": "NOT_ZIP",
        "MemberCount": 0,
        "UnsafeMembers": [],
        "CorruptMember": None,
    }
    if FilePath.suffix.lower() != ".zip":
        return Result
    try:
        with zipfile.ZipFile(FilePath) as Archive:
            Names = Archive.namelist()
            Result["MemberCount"] = len(Names)
            Result["UnsafeMembers"] = [Name for Name in Names if not IsSafeArchiveMember(Name)]
            Result["CorruptMember"] = Archive.testzip()
            if Result["UnsafeMembers"]:
                Result["Status"] = "UNSAFE_PATH"
            elif Result["CorruptMember"] is not None:
                Result["Status"] = "CRC_FAILURE"
            else:
                Result["Status"] = "OK"
    except (OSError, zipfile.BadZipFile) as Error:
        Result["Status"] = "BAD_ZIP"
        Result["Error"] = str(Error)
    return Result


def ReadCsvShape(FilePath: Path) -> dict[str, Any]:
    Result: dict[str, Any] = {
        "Path": str(FilePath),
        "Status": "UNREAD",
        "Rows": 0,
        "Columns": 0,
        "Header": [],
        "DuplicateRows": 0,
        "DuplicateIds": 0,
        "BlankCells": 0,
    }
    try:
        with FilePath.open("r", encoding="utf-8-sig", newline="") as FileHandle:
            Reader = csv.reader(FileHandle)
            AllRows = list(Reader)
        if not AllRows:
            Result["Status"] = "EMPTY"
            return Result
        Header = AllRows[0]
        Rows = AllRows[1:]
        Result["Status"] = "OK"
        Result["Rows"] = len(Rows)
        Result["Columns"] = len(Header)
        Result["Header"] = Header
        Result["RaggedRows"] = sum(1 for Row in Rows if len(Row) != len(Header))
        Result["DuplicateRows"] = len(Rows) - len(set(tuple(Row) for Row in Rows))
        Ids = [Row[0] for Row in Rows if Row]
        Result["DuplicateIds"] = len(Ids) - len(set(Ids))
        Result["BlankCells"] = sum(1 for Row in Rows for Cell in Row if Cell.strip() == "")
        if FilePath.name.lower() == "sample_submission.csv":
            Result["ExpectedHeaderMatch"] = Header == ExpectedSubmissionColumns
            Result["ExpectedDummyRowCountMatch"] = len(Rows) == 3
    except (OSError, UnicodeError, csv.Error) as Error:
        Result["Status"] = "PARSE_ERROR"
        Result["Error"] = str(Error)
    return Result


def ReadAudioInfo(FilePath: Path) -> dict[str, Any]:
    Result: dict[str, Any] = {
        "relative_path": "",
        "suffix": FilePath.suffix.lower(),
        "signature": DetectSignature(FilePath),
        "parse_status": "UNREAD",
        "duration_seconds": "",
        "sample_rate": "",
        "channels": "",
        "format": "",
        "subtype": "",
        "peak_abs": "",
        "rms": "",
        "silent_fraction": "",
        "clipped_fraction": "",
        "range_issues": "",
    }
    try:
        import numpy as Numpy
        import soundfile

        Info = soundfile.info(str(FilePath))
        DurationSeconds = float(Info.duration)
        RangeIssues: list[str] = []
        if DurationSeconds < 4.0 or DurationSeconds > 60.0:
            RangeIssues.append("DURATION_OUTSIDE_4_60")
        if int(Info.samplerate) != 16000:
            RangeIssues.append("SAMPLE_RATE_NOT_16000")
        if int(Info.channels) not in {1, 2}:
            RangeIssues.append("CHANNELS_NOT_MONO_STEREO")
        PeakAbsolute = 0.0
        SquareSum = 0.0
        SilentCount = 0
        ClippedCount = 0
        SampleValueCount = 0
        for AudioBlock in soundfile.blocks(
            str(FilePath),
            blocksize=65536,
            dtype="float32",
            always_2d=True,
        ):
            AbsoluteBlock = Numpy.abs(AudioBlock)
            BlockSampleCount = int(AbsoluteBlock.size)
            if BlockSampleCount == 0:
                continue
            PeakAbsolute = max(PeakAbsolute, float(AbsoluteBlock.max()))
            SquareSum += float(Numpy.square(AudioBlock, dtype=Numpy.float64).sum(dtype=Numpy.float64))
            SilentCount += int(Numpy.count_nonzero(AbsoluteBlock < 1.0e-4))
            ClippedCount += int(Numpy.count_nonzero(AbsoluteBlock >= 0.999))
            SampleValueCount += BlockSampleCount
        if SampleValueCount == 0:
            RootMeanSquare = 0.0
            SilentFraction = 0.0
            ClippedFraction = 0.0
            RangeIssues.append("NO_AUDIO_SAMPLES")
        else:
            RootMeanSquare = float(Numpy.sqrt(SquareSum / SampleValueCount))
            SilentFraction = SilentCount / SampleValueCount
            ClippedFraction = ClippedCount / SampleValueCount
        Result.update(
            {
                "parse_status": "OK",
                "duration_seconds": f"{DurationSeconds:.6f}",
                "sample_rate": int(Info.samplerate),
                "channels": int(Info.channels),
                "format": Info.format,
                "subtype": Info.subtype,
                "peak_abs": f"{PeakAbsolute:.9f}",
                "rms": f"{RootMeanSquare:.9f}",
                "silent_fraction": f"{SilentFraction:.9f}",
                "clipped_fraction": f"{ClippedFraction:.9f}",
                "range_issues": "|".join(RangeIssues),
            }
        )
    except ImportError:
        Result["parse_status"] = "SOUNDFILE_NOT_INSTALLED"
    except (OSError, RuntimeError, ValueError) as Error:
        Result["parse_status"] = "PARSE_ERROR"
        Result["range_issues"] = str(Error)
    return Result


def WriteCsv(FilePath: Path, FieldNames: list[str], Rows: list[dict[str, Any]]) -> None:
    with FilePath.open("w", encoding="utf-8-sig", newline="") as FileHandle:
        Writer = csv.DictWriter(FileHandle, fieldnames=FieldNames, extrasaction="ignore")
        Writer.writeheader()
        Writer.writerows(Rows)


def Main() -> int:
    Arguments = ParseArguments()
    RawDirectory = Arguments.RawDirectory.resolve()
    OutputDirectory = Arguments.OutputDirectory.resolve()
    OutputDirectory.mkdir(parents=True, exist_ok=True)
    AuditTime = datetime.now(timezone.utc).isoformat()

    ManifestRows: list[dict[str, Any]] = []
    AudioRows: list[dict[str, Any]] = []
    CsvResults: list[dict[str, Any]] = []
    ZipResults: list[dict[str, Any]] = []
    Issues: list[str] = []

    Files = sorted(PathItem for PathItem in RawDirectory.rglob("*") if PathItem.is_file()) if RawDirectory.exists() else []
    if not RawDirectory.exists():
        Issues.append("RAW_DIRECTORY_MISSING")
    if not Files:
        Issues.append("RAW_FILES_MISSING")

    HashCounts: Counter[str] = Counter()
    for FilePath in Files:
        RelativePath = FilePath.relative_to(RawDirectory).as_posix()
        FileHash = CalculateSha256(FilePath)
        HashCounts[FileHash] += 1
        ArchiveResult = AuditZip(FilePath)
        ParseStatus = "NOT_APPLICABLE"
        ArchiveStatus = ArchiveResult["Status"]
        if FilePath.suffix.lower() == ".csv":
            CsvResult = ReadCsvShape(FilePath)
            CsvResult["RelativePath"] = RelativePath
            CsvResults.append(CsvResult)
            ParseStatus = CsvResult["Status"]
        elif FilePath.suffix.lower() in AudioSuffixes:
            AudioResult = ReadAudioInfo(FilePath)
            AudioResult["relative_path"] = RelativePath
            AudioRows.append(AudioResult)
            ParseStatus = AudioResult["parse_status"]
        elif FilePath.suffix.lower() == ".zip":
            ZipResult = {"RelativePath": RelativePath, **ArchiveResult}
            ZipResults.append(ZipResult)
            ParseStatus = ArchiveResult["Status"]
        ManifestRows.append(
            {
                "relative_path": RelativePath,
                "role": "raw_file",
                "source_url": OfficialDataUrl if FilePath.name.lower() == "open.zip" else "UNREGISTERED_SOURCE",
                "retrieved_at_utc": "UNKNOWN",
                "bytes": FilePath.stat().st_size,
                "sha256": FileHash,
                "archive_status": ArchiveStatus,
                "parse_status": ParseStatus,
                "notes": "",
            }
        )

    for Row in ManifestRows:
        if HashCounts[str(Row["sha256"])] > 1:
            Row["notes"] = "DUPLICATE_BYTES"

    OpenZipPresent = any(FilePath.name.lower() == "open.zip" for FilePath in Files)
    SampleSubmissionPresent = any(FilePath.name.lower() == "sample_submission.csv" for FilePath in Files)
    DummyAudioCount = sum(1 for FilePath in Files if FilePath.suffix.lower() in AudioSuffixes)
    if not OpenZipPresent:
        Issues.append("OFFICIAL_OPEN_ZIP_MISSING")
    if not SampleSubmissionPresent:
        Issues.append("SAMPLE_SUBMISSION_NOT_EXTRACTED_OR_MISSING")
    if DummyAudioCount == 0:
        Issues.append("AUDIO_FILES_NOT_EXTRACTED_OR_MISSING")

    ManifestFields = [
        "relative_path",
        "role",
        "source_url",
        "retrieved_at_utc",
        "bytes",
        "sha256",
        "archive_status",
        "parse_status",
        "notes",
    ]
    AudioFields = [
        "relative_path",
        "suffix",
        "signature",
        "parse_status",
        "duration_seconds",
        "sample_rate",
        "channels",
        "format",
        "subtype",
        "peak_abs",
        "rms",
        "silent_fraction",
        "clipped_fraction",
        "range_issues",
    ]
    WriteCsv(OutputDirectory / "file-inventory.csv", ManifestFields, ManifestRows)
    WriteCsv(OutputDirectory / "audio-inventory.csv", AudioFields, AudioRows)

    Summary = {
        "DATA_READINESS": "BLOCKED",
        "AuditTimeUtc": AuditTime,
        "RawDirectory": str(RawDirectory),
        "RawDirectoryExists": RawDirectory.exists(),
        "FileCount": len(Files),
        "AudioFileCount": len(AudioRows),
        "CsvFileCount": len(CsvResults),
        "OpenZipPresent": OpenZipPresent,
        "SampleSubmissionPresent": SampleSubmissionPresent,
        "DuplicateByteGroups": sum(1 for Count in HashCounts.values() if Count > 1),
        "Issues": sorted(set(Issues)),
        "CsvResults": CsvResults,
        "ZipResults": ZipResults,
        "ReadinessNote": "Overall readiness remains BLOCKED until official dummy data and registered labeled training data are fully audited",
    }
    with (OutputDirectory / "audit-run.json").open("w", encoding="utf-8") as FileHandle:
        json.dump(Summary, FileHandle, ensure_ascii=False, indent=2)
        FileHandle.write("\n")

    json.dump(Summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 2 if Issues else 0


if __name__ == "__main__":
    raise SystemExit(Main())
