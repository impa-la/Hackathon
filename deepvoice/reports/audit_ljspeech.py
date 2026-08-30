#!/usr/bin/env python3
"""Read-only, reproducible audit of the locally extracted LJSpeech 1.1 corpus.

The script never extracts, rewrites, normalizes, or otherwise mutates source data.
It streams every WAV once and writes only beneath --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_ARCHIVE_SHA256 = "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5"
ID_RE = re.compile(r"^LJ\d{3}-\d{4}$")
SILENCE_THRESHOLD = 1e-4
CLIPPING_THRESHOLD = 0.999
BLOCK_FRAMES = 262_144


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def stable_text_key(text: str) -> str:
    canonical = " ".join(text.casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_metadata(path: Path):
    rows = []
    malformed = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        # LJSpeech is pipe-delimited plain text, not RFC 4180 CSV. Double quotes
        # occur as literal transcript characters and must not open quoted fields.
        for line_number, values in enumerate(
            csv.reader(stream, delimiter="|", quoting=csv.QUOTE_NONE), start=1
        ):
            if len(values) != 3:
                malformed.append({"line_number": line_number, "field_count": len(values)})
                continue
            clip_id, transcript, normalized = values
            rows.append(
                {
                    "line_number": line_number,
                    "id": clip_id,
                    "transcript": transcript,
                    "normalized": normalized,
                    "normalized_text_group_sha256": stable_text_key(normalized),
                }
            )
    return rows, malformed


def decode_pcm(raw: bytes, sample_width: int) -> tuple[np.ndarray, float]:
    """Decode little-endian integer PCM and return signed samples and full scale."""
    if sample_width == 1:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128, 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2"), 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        value = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        value = np.where(value & 0x800000, value - 0x1000000, value)
        return value, 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4"), 2_147_483_648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width}")


def audit_wav(path: Path) -> dict:
    file_digest = hashlib.sha256()
    with path.open("rb") as raw_stream:
        while block := raw_stream.read(8 * 1024 * 1024):
            file_digest.update(block)

    pcm_digest = hashlib.sha256()
    sample_count = 0
    sum_squares = 0.0
    silent_count = 0
    clipped_count = 0
    peak_abs = 0.0
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        compression_type = wav.getcomptype()
        compression_name = wav.getcompname()
        while raw := wav.readframes(BLOCK_FRAMES):
            pcm_digest.update(raw)
            samples, full_scale = decode_pcm(raw, sample_width)
            normalized = samples.astype(np.float64) / full_scale
            absolute = np.abs(normalized)
            sample_count += int(samples.size)
            sum_squares += float(np.dot(normalized, normalized))
            silent_count += int(np.count_nonzero(absolute < SILENCE_THRESHOLD))
            clipped_count += int(np.count_nonzero(absolute >= CLIPPING_THRESHOLD))
            if absolute.size:
                peak_abs = max(peak_abs, float(absolute.max()))

    expected_samples = frames * channels
    if sample_count != expected_samples:
        raise ValueError(f"decoded samples {sample_count} != header samples {expected_samples}")
    return {
        "file_sha256": file_digest.hexdigest(),
        "pcm_sha256": pcm_digest.hexdigest(),
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bits": sample_width * 8,
        "compression_type": compression_type,
        "compression_name": compression_name,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "sample_count": sample_count,
        "peak_abs_normalized": peak_abs,
        "rms_normalized": math.sqrt(sum_squares / sample_count) if sample_count else 0.0,
        "silent_sample_fraction_lt_1e_4": silent_count / sample_count if sample_count else 0.0,
        "clipped_sample_fraction_ge_0_999": clipped_count / sample_count if sample_count else 0.0,
        "silent_sample_count": silent_count,
        "clipped_sample_count": clipped_count,
    }


def quantiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std_population": float(arr.std()),
    }


def csv_write(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--wavefake-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    archive = args.archive.resolve()
    wavefake_root = args.wavefake_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_root / "metadata.csv"
    readme_path = dataset_root / "README"
    wav_dir = dataset_root / "wavs"
    source_state_before = {
        "archive": file_state(archive),
        "metadata": file_state(metadata_path),
        "readme": file_state(readme_path),
    }
    archive_sha256 = sha256_file(archive)
    metadata_sha256 = sha256_file(metadata_path)
    readme_sha256 = sha256_file(readme_path)

    metadata_rows, malformed_metadata_rows = load_metadata(metadata_path)
    metadata_by_id = {}
    duplicate_metadata_ids = Counter(row["id"] for row in metadata_rows)
    for row in metadata_rows:
        metadata_by_id.setdefault(row["id"], row)

    wav_paths = sorted(wav_dir.glob("*.wav"), key=lambda item: item.name)
    inventory = []
    parse_errors = []
    for index, wav_path in enumerate(wav_paths, start=1):
        clip_id = wav_path.stem
        meta = metadata_by_id.get(clip_id)
        row = {
            "id": clip_id,
            "relative_path": wav_path.relative_to(dataset_root).as_posix(),
            "file_bytes": wav_path.stat().st_size,
            "parse_status": "OK",
            "metadata_present": meta is not None,
            "transcript_chars": len(meta["transcript"]) if meta else "",
            "normalized_transcript_chars": len(meta["normalized"]) if meta else "",
            "normalized_text_group_sha256": meta["normalized_text_group_sha256"] if meta else "",
            "id_pattern_valid": bool(ID_RE.fullmatch(clip_id)),
            "issues": "",
        }
        try:
            row.update(audit_wav(wav_path))
            issues = []
            if row["compression_type"] != "NONE":
                issues.append(f"compression={row['compression_type']}")
            if row["channels"] != 1:
                issues.append(f"channels={row['channels']}")
            if row["sample_rate_hz"] != 22050:
                issues.append(f"sample_rate={row['sample_rate_hz']}")
            if row["sample_width_bits"] != 16:
                issues.append(f"bits={row['sample_width_bits']}")
            if not row["metadata_present"]:
                issues.append("metadata_missing")
            if not row["id_pattern_valid"]:
                issues.append("invalid_id_pattern")
            row["issues"] = ";".join(issues)
        except Exception as error:  # Keep a row for every file and report the error.
            row["parse_status"] = "ERROR"
            row["issues"] = f"{type(error).__name__}: {error}"
            parse_errors.append({"id": clip_id, "path": str(wav_path), "error": row["issues"]})
        inventory.append(row)
        if index % 1000 == 0:
            print(f"audited {index}/{len(wav_paths)} WAV files", flush=True)

    ok_rows = [row for row in inventory if row["parse_status"] == "OK"]
    wav_ids = {row["id"] for row in inventory}
    metadata_ids = set(metadata_by_id)
    missing_wav_for_metadata = sorted(metadata_ids - wav_ids)
    wav_without_metadata = sorted(wav_ids - metadata_ids)

    file_hash_groups = defaultdict(list)
    pcm_hash_groups = defaultdict(list)
    for row in ok_rows:
        file_hash_groups[row["file_sha256"]].append(row["id"])
        pcm_hash_groups[row["pcm_sha256"]].append(row["id"])
    duplicate_file_hash_groups = {key: ids for key, ids in file_hash_groups.items() if len(ids) > 1}
    duplicate_pcm_hash_groups = {key: ids for key, ids in pcm_hash_groups.items() if len(ids) > 1}

    transcript_groups = defaultdict(list)
    for row in metadata_rows:
        transcript_groups[row["normalized_text_group_sha256"]].append(row["id"])
    duplicate_transcript_groups = {key: ids for key, ids in transcript_groups.items() if len(ids) > 1}

    duration_values = [float(row["duration_seconds"]) for row in ok_rows]
    silence_values = [float(row["silent_sample_fraction_lt_1e_4"]) for row in ok_rows]
    clipping_values = [float(row["clipped_sample_fraction_ge_0_999"]) for row in ok_rows]
    rms_values = [float(row["rms_normalized"]) for row in ok_rows]
    peak_values = [float(row["peak_abs_normalized"]) for row in ok_rows]
    total_sample_count = sum(int(row["sample_count"]) for row in ok_rows)
    total_silent_sample_count = sum(int(row["silent_sample_count"]) for row in ok_rows)
    total_clipped_sample_count = sum(int(row["clipped_sample_count"]) for row in ok_rows)

    extracted_files = sorted(item for item in dataset_root.rglob("*") if item.is_file())
    extracted_dirs = sorted(item for item in dataset_root.rglob("*") if item.is_dir())
    extracted_bytes = sum(item.stat().st_size for item in extracted_files)

    wavefake_zip = wavefake_root / "generated_audio.zip"
    wavefake_datasheet = wavefake_root / "datasheet.pdf"
    wavefake_audio_files = sorted(
        item
        for item in wavefake_root.rglob("*")
        if item.is_file() and item.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    )
    wavefake_zip_state_before_hash = file_state(wavefake_zip) if wavefake_zip.exists() else None
    wavefake_zip_sha256 = sha256_file(wavefake_zip) if wavefake_zip.exists() else None
    wavefake_zip_state_after_hash = file_state(wavefake_zip) if wavefake_zip.exists() else None
    wavefake_state = {
        "root": str(wavefake_root),
        "generated_audio_zip_exists": wavefake_zip.exists(),
        "generated_audio_zip_size_bytes": wavefake_zip.stat().st_size if wavefake_zip.exists() else None,
        "generated_audio_zip_sha256": wavefake_zip_sha256,
        "generated_audio_zip_state_before_hash": wavefake_zip_state_before_hash,
        "generated_audio_zip_state_after_hash": wavefake_zip_state_after_hash,
        "generated_audio_zip_stable_during_hash": wavefake_zip_state_before_hash == wavefake_zip_state_after_hash,
        "datasheet_exists": wavefake_datasheet.exists(),
        "datasheet_size_bytes": wavefake_datasheet.stat().st_size if wavefake_datasheet.exists() else None,
        "datasheet_sha256": sha256_file(wavefake_datasheet) if wavefake_datasheet.exists() else None,
        "extracted_audio_file_count": len(wavefake_audio_files),
    }

    source_state_after = {
        "archive": file_state(archive),
        "metadata": file_state(metadata_path),
        "readme": file_state(readme_path),
    }
    source_state_unchanged = source_state_before == source_state_after

    header_combinations = Counter(
        (
            row["sample_rate_hz"],
            row["channels"],
            row["sample_width_bits"],
            row["compression_type"],
        )
        for row in ok_rows
    )
    chapter_prefix_counts = Counter(row["id"].split("-")[0] for row in inventory)
    non_ascii_transcripts = sum(
        1 for row in metadata_rows if any(ord(char) > 127 for char in row["transcript"])
    )
    exact_transcript_duplicate_groups = Counter(row["transcript"] for row in metadata_rows)
    exact_normalized_duplicate_groups = Counter(row["normalized"] for row in metadata_rows)

    if wavefake_state["extracted_audio_file_count"] > 0:
        wavefake_block = "WaveFake audio exists but real/fake ID pairing has not yet been audited."
    elif wavefake_state["generated_audio_zip_exists"] and not wavefake_state["generated_audio_zip_stable_during_hash"]:
        wavefake_block = "WaveFake generated_audio.zip was still changing during hashing and extracted audio count was zero."
    elif wavefake_state["generated_audio_zip_exists"] and wavefake_state["generated_audio_zip_size_bytes"] == 0:
        wavefake_block = "WaveFake generated_audio.zip was zero bytes and extracted audio count was zero."
    else:
        wavefake_block = "WaveFake extracted audio count was zero; archive completeness and pairing were not established."

    audit = {
        "data_readiness": "BLOCKED",
        "block_reason": f"{wavefake_block} Paired real/fake training rows cannot yet be constructed.",
        "method": {
            "scope": "all extracted LJSpeech WAV files; no sampling",
            "wav_block_frames": BLOCK_FRAMES,
            "silence_definition": "abs(sample / full_scale) < 1e-4",
            "clipping_definition": "abs(sample / full_scale) >= 0.999",
            "file_duplicate_key": "SHA-256 of complete WAV file bytes",
            "audio_duplicate_key": "SHA-256 of decoded PCM byte stream as stored in the WAV data chunks",
            "text_group_key": "SHA-256 of whitespace-collapsed, Unicode-preserving casefolded normalized transcript",
            "random_sampling": "none",
        },
        "source": {
            "dataset_root": str(dataset_root),
            "archive": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha256,
            "archive_sha256_expected": EXPECTED_ARCHIVE_SHA256,
            "archive_sha256_matches_expected": archive_sha256 == EXPECTED_ARCHIVE_SHA256,
            "metadata_sha256": metadata_sha256,
            "readme_sha256": readme_sha256,
            "source_state_unchanged_during_audit": source_state_unchanged,
            "extracted_member_count_including_root_children": len(extracted_files) + len(extracted_dirs),
            "extracted_file_count": len(extracted_files),
            "extracted_directory_count": len(extracted_dirs),
            "extracted_bytes": extracted_bytes,
        },
        "metadata": {
            "row_count_valid": len(metadata_rows),
            "malformed_row_count": len(malformed_metadata_rows),
            "malformed_rows": malformed_metadata_rows,
            "unique_id_count": len(metadata_ids),
            "duplicate_id_count": sum(count - 1 for count in duplicate_metadata_ids.values() if count > 1),
            "invalid_id_pattern_count": sum(not bool(ID_RE.fullmatch(row["id"])) for row in metadata_rows),
            "empty_transcript_count": sum(not row["transcript"].strip() for row in metadata_rows),
            "empty_normalized_transcript_count": sum(not row["normalized"].strip() for row in metadata_rows),
            "non_ascii_transcript_row_count": non_ascii_transcripts,
            "transcript_differs_from_normalized_count": sum(row["transcript"] != row["normalized"] for row in metadata_rows),
            "exact_transcript_duplicate_group_count": sum(count > 1 for count in exact_transcript_duplicate_groups.values()),
            "exact_transcript_duplicate_affected_rows": sum(count for count in exact_transcript_duplicate_groups.values() if count > 1),
            "exact_normalized_duplicate_group_count": sum(count > 1 for count in exact_normalized_duplicate_groups.values()),
            "exact_normalized_duplicate_affected_rows": sum(count for count in exact_normalized_duplicate_groups.values() if count > 1),
            "canonical_normalized_text_duplicate_group_count": len(duplicate_transcript_groups),
            "canonical_normalized_text_duplicate_affected_rows": sum(len(ids) for ids in duplicate_transcript_groups.values()),
        },
        "audio": {
            "wav_file_count": len(wav_paths),
            "wav_total_bytes": sum(path.stat().st_size for path in wav_paths),
            "parse_ok_count": len(ok_rows),
            "parse_error_count": len(parse_errors),
            "parse_errors": parse_errors,
            "metadata_joined_wav_count": sum(bool(row["metadata_present"]) for row in inventory),
            "wav_without_metadata_count": len(wav_without_metadata),
            "wav_without_metadata_ids": wav_without_metadata,
            "metadata_without_wav_count": len(missing_wav_for_metadata),
            "metadata_without_wav_ids": missing_wav_for_metadata,
            "header_combinations": [
                {
                    "sample_rate_hz": key[0],
                    "channels": key[1],
                    "sample_width_bits": key[2],
                    "compression_type": key[3],
                    "count": count,
                }
                for key, count in sorted(header_combinations.items())
            ],
            "total_frames": sum(int(row["frames"]) for row in ok_rows),
            "total_sample_count": total_sample_count,
            "total_duration_seconds": sum(duration_values),
            "total_duration_hours": sum(duration_values) / 3600,
            "duration_seconds_distribution": quantiles(duration_values),
            "silence_fraction_distribution": quantiles(silence_values),
            "clipping_fraction_distribution": quantiles(clipping_values),
            "rms_distribution": quantiles(rms_values),
            "peak_abs_distribution": quantiles(peak_values),
            "zero_length_count": sum(float(row["duration_seconds"]) == 0 for row in ok_rows),
            "all_silent_count": sum(float(row["silent_sample_fraction_lt_1e_4"]) == 1.0 for row in ok_rows),
            "total_silent_sample_count": total_silent_sample_count,
            "overall_silent_sample_fraction_lt_1e_4": total_silent_sample_count / total_sample_count,
            "total_clipped_sample_count": total_clipped_sample_count,
            "overall_clipped_sample_fraction_ge_0_999": total_clipped_sample_count / total_sample_count,
            "any_clipped_sample_file_count": sum(int(row["clipped_sample_count"]) > 0 for row in ok_rows),
            "clipped_file_ids": [row["id"] for row in ok_rows if int(row["clipped_sample_count"]) > 0],
            "silence_fraction_ge_50pct_count": sum(float(row["silent_sample_fraction_lt_1e_4"]) >= 0.5 for row in ok_rows),
            "duration_gt_10_seconds_count": sum(float(row["duration_seconds"]) > 10.0 for row in ok_rows),
            "duration_lt_1_second_count": sum(float(row["duration_seconds"]) < 1.0 for row in ok_rows),
            "file_hash_duplicate_group_count": len(duplicate_file_hash_groups),
            "file_hash_duplicate_affected_count": sum(len(ids) for ids in duplicate_file_hash_groups.values()),
            "pcm_hash_duplicate_group_count": len(duplicate_pcm_hash_groups),
            "pcm_hash_duplicate_affected_count": sum(len(ids) for ids in duplicate_pcm_hash_groups.values()),
        },
        "id_structure": {
            "prefix_count": len(chapter_prefix_counts),
            "prefix_clip_counts": dict(sorted(chapter_prefix_counts.items())),
        },
        "wavefake": wavefake_state,
        "local_document_evidence": {
            "ljspeech_readme_version": "1.1",
            "ljspeech_readme_public_domain_statement": True,
            "ljspeech_readme_original_recording_codec": "128 kbps MP3",
            "ljspeech_readme_mp3_artifact_warning": True,
            "wavefake_datasheet_reported_generated_clip_count": 117985,
            "wavefake_datasheet_reported_ljspeech_based_network_count": 7,
            "wavefake_datasheet_no_recommended_split": True,
            "wavefake_datasheet_reference_audio_not_redistributed": True,
        },
    }

    inventory_fields = [
        "id", "relative_path", "file_bytes", "file_sha256", "pcm_sha256",
        "parse_status", "channels", "sample_rate_hz", "sample_width_bits",
        "compression_type", "compression_name", "frames", "duration_seconds",
        "sample_count", "peak_abs_normalized", "rms_normalized",
        "silent_sample_fraction_lt_1e_4", "clipped_sample_fraction_ge_0_999",
        "silent_sample_count", "clipped_sample_count", "metadata_present",
        "transcript_chars", "normalized_transcript_chars",
        "normalized_text_group_sha256", "id_pattern_valid", "issues",
    ]
    csv_write(output_dir / "ljspeech-audio-inventory.csv", inventory, inventory_fields)

    duplicate_rows = []
    for duplicate_type, groups in (
        ("file_sha256", duplicate_file_hash_groups),
        ("pcm_sha256", duplicate_pcm_hash_groups),
        ("normalized_text_group_sha256", duplicate_transcript_groups),
    ):
        for digest, ids in sorted(groups.items()):
            duplicate_rows.append(
                {
                    "duplicate_type": duplicate_type,
                    "sha256": digest,
                    "member_count": len(ids),
                    "ids": ";".join(ids),
                }
            )
    csv_write(
        output_dir / "ljspeech-duplicate-groups.csv",
        duplicate_rows,
        ["duplicate_type", "sha256", "member_count", "ids"],
    )

    with (output_dir / "ljspeech-audit-run.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(json.dumps({
        "wav_count": len(wav_paths),
        "parse_error_count": len(parse_errors),
        "duration_hours": audit["audio"]["total_duration_hours"],
        "archive_sha256_matches_expected": audit["source"]["archive_sha256_matches_expected"],
        "source_state_unchanged": source_state_unchanged,
        "wavefake_audio_count": wavefake_state["extracted_audio_file_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
