"""Read-only, reproducible audit of DACON DeepVoice open.zip.

The source archive is never modified or extracted beside itself. The nested
baseline archive is copied to a temporary directory only long enough to verify
every member's CRC and SHA-256, then the temporary copy is deleted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import struct
import tempfile
import wave
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


CHUNK_BYTES = 8 * 1024 * 1024
EXPECTED_SUBMISSION_COLUMNS = [
    "ID",
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_dir", type=Path, required=True)
    return parser.parse_args()


def stream_sha256(handle: BinaryIO, copy_to: BinaryIO | None = None) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    total = 0
    prefix = bytearray()
    while True:
        block = handle.read(CHUNK_BYTES)
        if not block:
            break
        if len(prefix) < 64:
            prefix.extend(block[: 64 - len(prefix)])
        digest.update(block)
        total += len(block)
        if copy_to is not None:
            copy_to.write(block)
    return digest.hexdigest(), total, bytes(prefix)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return stream_sha256(handle)[0]


def safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and not (candidate.parts and ":" in candidate.parts[0])
    )


def signature(prefix: bytes) -> str:
    if not prefix:
        return "EMPTY"
    if prefix.startswith(b"PK\x03\x04") or prefix.startswith(b"PK\x05\x06"):
        return "ZIP"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE":
        return "WAV"
    if prefix.startswith(b"fLaC"):
        return "FLAC"
    if prefix.startswith(b"ID3"):
        return "MP3_ID3"
    if len(prefix) >= 2 and prefix[0] == 0xFF and prefix[1] & 0xE0 == 0xE0:
        return "MP3_FRAME"
    if prefix.startswith(b"\x80"):
        return "PYTHON_PICKLE_OR_TORCH"
    if prefix.startswith(b"{\n") or prefix.startswith(b"{"):
        return "JSON_TEXT"
    text = prefix.decode("utf-8-sig", errors="ignore")
    printable = sum(character.isprintable() or character in "\r\n\t" for character in text)
    if not text or printable / len(text) < 0.85:
        return "BINARY_UNKNOWN"
    if "," in text or text.strip().startswith(("import ", "from ", "#", "-")):
        return "UTF8_TEXT"
    return "UTF8_TEXT_OR_EMPTY"


def extension_signature_match(name: str, detected: str) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    expected = {
        ".zip": {"ZIP"},
        ".wav": {"WAV"},
        ".flac": {"FLAC"},
        ".mp3": {"MP3_ID3", "MP3_FRAME"},
        ".csv": {"UTF8_TEXT", "UTF8_TEXT_OR_EMPTY"},
        ".py": {"EMPTY", "UTF8_TEXT", "UTF8_TEXT_OR_EMPTY"},
        ".txt": {"EMPTY", "UTF8_TEXT", "UTF8_TEXT_OR_EMPTY"},
        ".json": {"JSON_TEXT", "UTF8_TEXT", "UTF8_TEXT_OR_EMPTY"},
        ".md": {"UTF8_TEXT", "UTF8_TEXT_OR_EMPTY"},
        ".pth": {"ZIP", "PYTHON_PICKLE_OR_TORCH", "BINARY_UNKNOWN"},
        ".pt": {"ZIP", "PYTHON_PICKLE_OR_TORCH", "BINARY_UNKNOWN"},
        # Modern torch.save/Hugging Face weights can be ZIP containers even
        # when the conventional extension is .bin.
        ".bin": {"ZIP", "BINARY_UNKNOWN", "PYTHON_PICKLE_OR_TORCH"},
    }
    if suffix not in expected:
        return "NO_SIGNATURE_RULE"
    return "MATCH" if detected in expected[suffix] else "MISMATCH"


def compression_name(value: int) -> str:
    return {
        zipfile.ZIP_STORED: "STORED",
        zipfile.ZIP_DEFLATED: "DEFLATED",
        zipfile.ZIP_BZIP2: "BZIP2",
        zipfile.ZIP_LZMA: "LZMA",
    }.get(value, f"UNKNOWN_{value}")


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_submission(data: bytes, archive_path: str, submission_expected: bool = True) -> dict:
    result = {
        "archive_path": archive_path,
        "csv_kind": "submission_format" if submission_expected else "auxiliary_model_metadata",
        "encoding": "utf-8-sig",
        "parse_status": "UNREAD",
        "rows": 0,
        "columns": 0,
        "header": "",
        "duplicate_rows": 0,
        "duplicate_ids": 0,
        "blank_cells": 0,
        "ragged_rows": 0,
        "probability_non_numeric": 0,
        "probability_out_of_range": 0,
        "constant_columns": "",
        "target_columns_present": 0,
        "expected_header_match": False,
    }
    try:
        text = data.decode("utf-8-sig")
        table = list(csv.reader(io.StringIO(text)))
        if not table:
            result["parse_status"] = "EMPTY"
            return result
        header, rows = table[0], table[1:]
        width = len(header)
        duplicate_rows = len(rows) - len(set(map(tuple, rows)))
        ids = [row[0] for row in rows if row]
        non_numeric = 0
        out_of_range = 0
        if submission_expected:
            for row in rows:
                for cell in row[1:]:
                    try:
                        value = float(cell)
                    except ValueError:
                        non_numeric += 1
                        continue
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        out_of_range += 1
        constants = []
        for index, column in enumerate(header):
            values = [row[index] for row in rows if len(row) > index]
            if values and len(set(values)) == 1:
                constants.append(column)
        result.update(
            {
                "parse_status": "OK",
                "rows": len(rows),
                "columns": width,
                "header": "|".join(header),
                "duplicate_rows": duplicate_rows,
                "duplicate_ids": len(ids) - len(set(ids)),
                "blank_cells": sum(cell.strip() == "" for row in rows for cell in row),
                "ragged_rows": sum(len(row) != width for row in rows),
                "probability_non_numeric": non_numeric if submission_expected else "NOT_APPLICABLE",
                "probability_out_of_range": out_of_range if submission_expected else "NOT_APPLICABLE",
                "constant_columns": "|".join(constants),
                "target_columns_present": sum("LABEL" in x.upper() or "TARGET" in x.upper() for x in header),
                "expected_header_match": header == EXPECTED_SUBMISSION_COLUMNS if submission_expected else "NOT_APPLICABLE",
            }
        )
    except (UnicodeError, csv.Error) as error:
        result["parse_status"] = f"ERROR: {error}"
    return result


def read_wav(data: bytes, archive_path: str) -> dict:
    result = {
        "archive_path": archive_path,
        "parse_status": "UNREAD",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "codec": "",
        "channels": "",
        "sample_rate": "",
        "sample_width_bytes": "",
        "frames": "",
        "duration_seconds": "",
        "peak_abs": "",
        "rms": "",
        "silent_fraction_lt_1e-4": "",
        "clipped_fraction_ge_0_999": "",
        "duration_in_official_4_60_range": "",
    }
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            width = wav.getsampwidth()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
            codec = wav.getcomptype()
        if width != 2:
            raise ValueError(f"Only 16-bit PCM metrics implemented, got sample width {width}")
        values = struct.unpack(f"<{len(raw) // 2}h", raw)
        scale = 32768.0
        absolute = [abs(value) / scale for value in values]
        duration = frames / rate if rate else 0.0
        result.update(
            {
                "parse_status": "OK",
                "codec": f"WAV_{codec}_PCM_S16LE",
                "channels": channels,
                "sample_rate": rate,
                "sample_width_bytes": width,
                "frames": frames,
                "duration_seconds": f"{duration:.6f}",
                "peak_abs": f"{max(absolute, default=0.0):.9f}",
                "rms": f"{math.sqrt(sum(v * v for v in values) / len(values)) / scale:.9f}" if values else "0",
                "silent_fraction_lt_1e-4": f"{sum(v < 1.0e-4 for v in absolute) / len(absolute):.9f}" if absolute else "0",
                "clipped_fraction_ge_0_999": f"{sum(v >= 0.999 for v in absolute) / len(absolute):.9f}" if absolute else "0",
                "duration_in_official_4_60_range": 4.0 <= duration <= 60.0,
            }
        )
    except (wave.Error, EOFError, ValueError) as error:
        result["parse_status"] = f"ERROR: {error}"
    return result


def make_figures(output_dir: Path, outer_rows: list[dict], audio_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    file_rows = [row for row in outer_rows if row["is_dir"] == 0]
    labels = [PurePosixPath(row["member_path"]).name for row in file_rows]
    sizes_gib = [row["uncompressed_bytes"] / (1024**3) for row in file_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, sizes_gib, color=["#36558F"] + ["#75B9BE"] * (len(labels) - 1))
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_ylabel("Uncompressed size (GiB, symlog)")
    ax.set_title("open.zip member-size composition")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures / "zip-member-sizes.png", dpi=160)
    plt.close(fig)

    names = [PurePosixPath(row["archive_path"]).name for row in audio_rows]
    durations = [float(row["duration_seconds"]) for row in audio_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(names, durations, color="#F18F01")
    ax.axhline(4.0, color="#C73E1D", linestyle="--", label="Official evaluation minimum (4 s)")
    ax.set_ylim(0, max(4.5, max(durations, default=0) * 1.2))
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Dummy WAV durations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "dummy-audio-duration.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    zip_path = args.zip_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_time = datetime.now(timezone.utc).isoformat()
    source_stat_before = zip_path.stat()

    outer_sha = file_sha256(zip_path)
    outer_rows: list[dict] = []
    inner_rows: list[dict] = []
    audio_rows: list[dict] = []
    csv_rows: list[dict] = []
    issues: list[str] = []
    baseline_sha = ""
    baseline_crc_status = "NOT_TESTED"
    outer_crc_status = "OK"
    inner_crc_status = "NOT_APPLICABLE"
    inner_small_files: dict[str, bytes] = {}
    declared_sha_status = "NOT_PRESENT"
    declared_sha_entries = 0
    declared_sha_mismatches: list[str] = []

    temp_parent = output_dir / ".work"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nested-zip-", dir=temp_parent) as temp_name:
        nested_path = Path(temp_name) / "baseline_submit.zip"
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/")
                row = {
                    "archive_scope": "outer",
                    "member_path": normalized,
                    "is_dir": int(info.is_dir()),
                    "safe_path": int(safe_member(normalized)),
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "compression": compression_name(info.compress_type),
                    "crc32_expected": f"{info.CRC:08x}",
                    "sha256": "",
                    "signature": "DIRECTORY" if info.is_dir() else "",
                    "extension_signature": "NOT_APPLICABLE" if info.is_dir() else "",
                    "read_crc_status": "NOT_APPLICABLE" if info.is_dir() else "UNREAD",
                }
                if not safe_member(normalized):
                    issues.append(f"UNSAFE_OUTER_PATH:{normalized}")
                if not info.is_dir():
                    try:
                        with archive.open(info) as source:
                            if normalized == "baseline_submit.zip":
                                with nested_path.open("wb") as target:
                                    digest, count, prefix = stream_sha256(source, target)
                                baseline_sha = digest
                                baseline_crc_status = "OK"
                            else:
                                data = source.read()
                                digest = hashlib.sha256(data).hexdigest()
                                count = len(data)
                                prefix = data[:64]
                                if normalized.lower().endswith(".csv"):
                                    csv_rows.append(read_submission(data, f"outer::{normalized}"))
                                if normalized.lower().endswith(".wav"):
                                    audio_rows.append(read_wav(data, f"outer::{normalized}"))
                        detected = signature(prefix)
                        row.update(
                            {
                                "sha256": digest,
                                "signature": detected,
                                "extension_signature": extension_signature_match(normalized, detected),
                                "read_crc_status": "OK" if count == info.file_size else "SIZE_MISMATCH",
                            }
                        )
                    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                        row["read_crc_status"] = f"ERROR:{error}"
                        outer_crc_status = "FAILED"
                        issues.append(f"OUTER_MEMBER_READ_FAILURE:{normalized}:{error}")
                outer_rows.append(row)

        if nested_path.exists():
            try:
                with zipfile.ZipFile(nested_path) as nested:
                    inner_crc_status = "OK"
                    for info in nested.infolist():
                        normalized = info.filename.replace("\\", "/")
                        row = {
                            "archive_scope": "baseline_submit.zip",
                            "member_path": normalized,
                            "is_dir": int(info.is_dir()),
                            "safe_path": int(safe_member(normalized)),
                            "compressed_bytes": info.compress_size,
                            "uncompressed_bytes": info.file_size,
                            "compression": compression_name(info.compress_type),
                            "crc32_expected": f"{info.CRC:08x}",
                            "sha256": "",
                            "signature": "DIRECTORY" if info.is_dir() else "",
                            "extension_signature": "NOT_APPLICABLE" if info.is_dir() else "",
                            "read_crc_status": "NOT_APPLICABLE" if info.is_dir() else "UNREAD",
                        }
                        if not safe_member(normalized):
                            issues.append(f"UNSAFE_INNER_PATH:{normalized}")
                        if not info.is_dir():
                            try:
                                with nested.open(info) as source:
                                    if info.file_size <= 1024 * 1024:
                                        data = source.read()
                                        digest = hashlib.sha256(data).hexdigest()
                                        count = len(data)
                                        prefix = data[:64]
                                        inner_small_files[normalized] = data
                                        if normalized.lower().endswith(".csv"):
                                            csv_rows.append(
                                                read_submission(
                                                    data,
                                                    f"baseline_submit.zip::{normalized}",
                                                    submission_expected=False,
                                                )
                                            )
                                    else:
                                        digest, count, prefix = stream_sha256(source)
                                detected = signature(prefix)
                                row.update(
                                    {
                                        "sha256": digest,
                                        "signature": detected,
                                        "extension_signature": extension_signature_match(normalized, detected),
                                        "read_crc_status": "OK" if count == info.file_size else "SIZE_MISMATCH",
                                    }
                                )
                            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                                row["read_crc_status"] = f"ERROR:{error}"
                                inner_crc_status = "FAILED"
                                issues.append(f"INNER_MEMBER_READ_FAILURE:{normalized}:{error}")
                        inner_rows.append(row)
                    declared_path = "model/SHA256SUMS.txt"
                    if declared_path in inner_small_files:
                        declared_sha_status = "OK"
                        hash_by_path = {
                            row["member_path"]: row["sha256"]
                            for row in inner_rows
                            if row["is_dir"] == 0
                        }
                        text = inner_small_files[declared_path].decode("utf-8-sig")
                        for line in text.splitlines():
                            if not line.strip():
                                continue
                            parts = line.split(maxsplit=1)
                            if len(parts) != 2:
                                declared_sha_mismatches.append(f"MALFORMED:{line}")
                                continue
                            expected, member = parts
                            member = member.lstrip("* ").replace("\\", "/")
                            declared_sha_entries += 1
                            actual = hash_by_path.get(member)
                            if actual is None:
                                actual = hash_by_path.get(f"model/{member}")
                            if actual != expected.lower():
                                declared_sha_mismatches.append(
                                    f"{member}:expected={expected.lower()}:actual={actual}"
                                )
                        if declared_sha_mismatches:
                            declared_sha_status = "MISMATCH"
            except zipfile.BadZipFile as error:
                inner_crc_status = f"BAD_ZIP:{error}"
                issues.append(f"INNER_BAD_ZIP:{error}")
        else:
            issues.append("BASELINE_NESTED_ZIP_MISSING")

    try:
        temp_parent.rmdir()
    except OSError:
        pass

    all_file_rows = [row for row in outer_rows + inner_rows if row["is_dir"] == 0]
    sha_groups: dict[str, list[str]] = defaultdict(list)
    for row in all_file_rows:
        sha_groups[row["sha256"]].append(f"{row['archive_scope']}::{row['member_path']}")
    duplicate_rows = [
        {"sha256": digest, "count": len(paths), "paths": "|".join(paths)}
        for digest, paths in sha_groups.items()
        if digest and len(paths) > 1
    ]

    inventory_fields = [
        "archive_scope",
        "member_path",
        "is_dir",
        "safe_path",
        "compressed_bytes",
        "uncompressed_bytes",
        "compression",
        "crc32_expected",
        "sha256",
        "signature",
        "extension_signature",
        "read_crc_status",
    ]
    write_csv(output_dir / "zip-member-inventory.csv", outer_rows + inner_rows, inventory_fields)
    write_csv(
        output_dir / "audio-inventory.csv",
        audio_rows,
        [
            "archive_path",
            "parse_status",
            "bytes",
            "sha256",
            "codec",
            "channels",
            "sample_rate",
            "sample_width_bytes",
            "frames",
            "duration_seconds",
            "peak_abs",
            "rms",
            "silent_fraction_lt_1e-4",
            "clipped_fraction_ge_0_999",
            "duration_in_official_4_60_range",
        ],
    )
    write_csv(
        output_dir / "csv-audit.csv",
        csv_rows,
        [
            "archive_path",
            "csv_kind",
            "encoding",
            "parse_status",
            "rows",
            "columns",
            "header",
            "duplicate_rows",
            "duplicate_ids",
            "blank_cells",
            "ragged_rows",
            "probability_non_numeric",
            "probability_out_of_range",
            "constant_columns",
            "target_columns_present",
            "expected_header_match",
        ],
    )
    write_csv(output_dir / "duplicate-groups.csv", duplicate_rows, ["sha256", "count", "paths"])
    write_csv(
        output_dir / "data-manifest.csv",
        [
            {
                "relative_path": "deepvoice/open.zip",
                "role": "official_format_example_package_not_training_data",
                "source_url": "https://dacon.io/competitions/official/236749/data",
                "retrieved_at_utc": "USER_PROVIDED; exact download time unknown",
                "bytes": zip_path.stat().st_size,
                "sha256": outer_sha,
                "archive_status": outer_crc_status,
                "parse_status": "OK" if not issues else "PARTIAL_OR_FAILED",
                "notes": "Read in place; source not modified. Official page says no training dataset is provided.",
            }
        ],
        [
            "relative_path",
            "role",
            "source_url",
            "retrieved_at_utc",
            "bytes",
            "sha256",
            "archive_status",
            "parse_status",
            "notes",
        ],
    )

    make_figures(output_dir, outer_rows, audio_rows)

    extension_mismatches = [
        f"{row['archive_scope']}::{row['member_path']}"
        for row in all_file_rows
        if row["extension_signature"] == "MISMATCH"
    ]
    source_stat_after = zip_path.stat()
    source_modified = (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
    )
    summary = {
        "DATA_READINESS": "BLOCKED",
        "audit_time_utc": audit_time,
        "source_zip": str(zip_path),
        "source_bytes": source_stat_after.st_size,
        "source_sha256": outer_sha,
        "source_mtime_utc": datetime.fromtimestamp(source_stat_after.st_mtime, timezone.utc).isoformat(),
        "source_modified": source_modified,
        "zip_character": "OFFICIAL_FORMAT_EXAMPLE_AND_BASELINE_SUBMISSION; NOT_LABELED_TRAINING_DATA",
        "outer_member_count": len(outer_rows),
        "outer_file_count": sum(row["is_dir"] == 0 for row in outer_rows),
        "inner_member_count": len(inner_rows),
        "inner_file_count": sum(row["is_dir"] == 0 for row in inner_rows),
        "outer_crc_status": outer_crc_status,
        "baseline_member_sha256": baseline_sha,
        "baseline_member_crc_status": baseline_crc_status,
        "inner_crc_status": inner_crc_status,
        "declared_model_sha_status": declared_sha_status,
        "declared_model_sha_entries": declared_sha_entries,
        "declared_model_sha_mismatches": declared_sha_mismatches,
        "unsafe_path_count": sum(row["safe_path"] == 0 for row in outer_rows + inner_rows),
        "extension_signature_mismatch_count": len(extension_mismatches),
        "extension_signature_mismatches": extension_mismatches,
        "audio_file_count": len(audio_rows),
        "audio_unique_sha256_count": len({row["sha256"] for row in audio_rows}),
        "audio_total_duration_seconds": sum(float(row["duration_seconds"]) for row in audio_rows if row["parse_status"] == "OK"),
        "audio_duration_outside_official_range_count": sum(row["duration_in_official_4_60_range"] is False for row in audio_rows),
        "csv_file_count": len(csv_rows),
        "labeled_training_rows": 0,
        "target_columns_present": sum(int(row["target_columns_present"]) for row in csv_rows),
        "duplicate_byte_group_count": len(duplicate_rows),
        "issues": issues,
        "blocking_reasons": [
            "Official open.zip explicitly contains only a reference baseline, three dummy inputs, and sample_submission.csv.",
            "No labeled training samples or training-source registry are present in this archive.",
            "Class distribution, labeled duplicate leakage, train/validation shift, and lineage-safe splitting cannot be audited without the actual participant-constructed training corpus.",
        ],
        "content_sampling": {
            "seed": None,
            "method": "No sampling: all 3 audio files and every outer/inner archive member were read in full.",
            "limitations": "The three WAV files are duplicated format-check dummies and do not represent the private 1,200-file evaluation distribution.",
        },
    }
    with (output_dir / "audit-run.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
