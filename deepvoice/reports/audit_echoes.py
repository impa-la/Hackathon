#!/usr/bin/env python3
"""Read-only, full-file audit of the pinned Echoes dataset revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np


EXPECTED_SHA256 = "8746dcb367f2f547399201d442ffab9121c36415815947ed4784e29b60e25b59"
EXPECTED_ARCHIVE_BYTES = 8_598_345_242
EXPECTED_MEMBER_COUNT = 4_509
EXPECTED_FILE_COUNT = 4_489
EXPECTED_UNCOMPRESSED_BYTES = 8_843_084_266
OPEN_MODEL_PROVIDERS = {"acestep", "audioldm", "diffrhythm", "musicgen", "songgen"}
COMMERCIAL_PROVIDERS = {"brev", "elevenlabs", "mubert", "producer", "stableaudio", "suno", "udio"}
SILENCE_THRESHOLD = 1e-4
CLIPPING_THRESHOLD = 0.999
READ_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fma-metadata-root", type=Path, required=True)
    parser.add_argument("--fma-allowlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def file_state(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(READ_BYTES):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name: str) -> bool:
    candidate = PurePosixPath(name.replace("\\", "/"))
    return not candidate.is_absolute() and ".." not in candidate.parts and not (
        candidate.parts and ":" in candidate.parts[0]
    )


def normalize_path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


def normalize_reference(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"path_in_dataset", "original_audio", "generator", "type", "genre", "description", "duration"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected manifest fields: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            row = dict(row)
            row["manifest_row_number"] = row_number
            row["path_in_dataset"] = normalize_path(row["path_in_dataset"])
            rows.append(row)
    return rows


def load_curated_ids(tracks_csv: Path) -> set[int]:
    with tracks_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader); next(reader); next(reader)
        return {int(row[0]) for row in reader}


def load_fma_references(raw_csv: Path, curated_ids: set[int], allowlist_path: Path):
    allowlist = {}
    with allowlist_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            allowlist[int(row["track_id"])] = row
    exact = defaultdict(list)
    normalized = defaultdict(list)
    metadata = {}
    with raw_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            track_id = int(row["track_id"])
            if track_id not in curated_ids:
                continue
            display = f"{row['track_title']} - {row['artist_name']}"
            record = {
                "track_id": track_id,
                "display": display,
                "artist_id": row["artist_id"],
                "album_id": row["album_id"],
                "artist_name": row["artist_name"],
                "track_title": row["track_title"],
                "license_title": row["license_title"],
                "license_url": row["license_url"],
                "strict_allowlist": track_id in allowlist,
                "strict_license_category": allowlist.get(track_id, {}).get("license_category", "OUTSIDE_STRICT_ALLOWLIST"),
            }
            metadata[track_id] = record
            exact[display].append(track_id)
            normalized[normalize_reference(display)].append(track_id)
    return exact, normalized, metadata


def match_reference(value: str, exact: dict, normalized: dict, metadata: dict) -> dict:
    exact_ids = exact.get(value, [])
    if len(exact_ids) == 1:
        ids, status = exact_ids, "EXACT_UNIQUE"
    elif len(exact_ids) > 1:
        ids, status = exact_ids, "EXACT_AMBIGUOUS"
    else:
        ids = normalized.get(normalize_reference(value), [])
        status = "NORMALIZED_UNIQUE" if len(ids) == 1 else "NORMALIZED_AMBIGUOUS" if len(ids) > 1 else "UNMATCHED"
    if len(ids) == 1:
        row = metadata[ids[0]]
        return {
            "reference_match_status": status,
            "reference_fma_track_id": ids[0],
            "reference_fma_candidate_ids": str(ids[0]),
            "reference_artist_id": row["artist_id"],
            "reference_album_id": row["album_id"],
            "reference_strict_allowlist": row["strict_allowlist"],
            "reference_license_category": row["strict_license_category"],
            "semantic_pair_group": f"fma:{ids[0]}",
        }
    digest = hashlib.sha256(normalize_reference(value).encode("utf-8")).hexdigest()[:16]
    return {
        "reference_match_status": status,
        "reference_fma_track_id": "",
        "reference_fma_candidate_ids": ";".join(str(item) for item in sorted(ids)),
        "reference_artist_id": "",
        "reference_album_id": "",
        "reference_strict_allowlist": "",
        "reference_license_category": "",
        "semantic_pair_group": f"unresolved:{digest}",
    }


def provider_rights(provider: str) -> dict:
    if provider in OPEN_MODEL_PROVIDERS:
        return {
            "provider_license_tier": "OPEN_MODEL_FAMILY_EXACT_REVISION_UNVERIFIED",
            "evidence_status": "CONDITIONAL_MODEL_VERSION_AND_OUTPUT_LINEAGE_NOT_PROVEN",
            "conservative_training_eligibility": "CONDITIONAL_HOLD",
            "training_eligible": False,
        }
    if provider in COMMERCIAL_PROVIDERS:
        return {
            "provider_license_tier": "COMMERCIAL_ACCOUNT_TIER_UNKNOWN",
            "evidence_status": "HOLD_GENERATION_TIER_AND_OUTPUT_RIGHTS_NOT_PROVEN",
            "conservative_training_eligibility": "HOLD",
            "training_eligible": False,
        }
    return {
        "provider_license_tier": "UNKNOWN_PROVIDER",
        "evidence_status": "HOLD_NO_PROVIDER_RIGHTS_EVIDENCE",
        "conservative_training_eligibility": "HOLD",
        "training_eligible": False,
    }


def mp3_frame_length_at(data: bytes, index: int) -> int | None:
    if index + 4 > len(data) or data[index] != 0xFF or (data[index + 1] & 0xE0) != 0xE0:
        return None
    version = (data[index + 1] >> 3) & 3
    layer = (data[index + 1] >> 1) & 3
    bitrate_index = (data[index + 2] >> 4) & 15
    rate_index = (data[index + 2] >> 2) & 3
    padding = (data[index + 2] >> 1) & 1
    if version == 1 or layer != 1 or bitrate_index in {0, 15} or rate_index == 3:
        return None
    rates = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
    b1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)
    b2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
    rate = rates[version][rate_index]
    bitrate = (b1 if version == 3 else b2)[bitrate_index]
    return int((144000 if version == 3 else 72000) * bitrate / rate) + padding


def inspect_signature(path: Path) -> dict:
    with path.open("rb") as stream:
        head = stream.read(min(path.stat().st_size, 2 * 1024 * 1024))
    extension = path.suffix.lower()
    if extension == ".wav":
        ok = len(head) >= 12 and head[:4] in {b"RIFF", b"RF64"} and head[8:12] == b"WAVE"
        return {"signature_status": "WAVE_VALIDATED" if ok else "WAVE_SIGNATURE_INVALID", "id3v2_signature": False}
    id3 = head.startswith(b"ID3")
    offset = 0
    if id3 and len(head) >= 10 and all(value < 128 for value in head[6:10]):
        offset = 10 + sum(value << shift for value, shift in zip(head[6:10], (21, 14, 7, 0)))
    found = False
    for index in range(min(offset, len(head)), len(head) - 4):
        length = mp3_frame_length_at(head, index)
        if length and mp3_frame_length_at(head, index + length):
            found = True
            break
    return {"signature_status": "MP3_FRAMES_VALIDATED" if found else "MP3_SIGNATURE_INVALID", "id3v2_signature": id3}


def parse_ffmpeg(stderr: str) -> dict:
    duration = None
    container_bitrate = None
    if match := re.search(r"Duration:\s*(\d+):(\d+):([0-9.]+),\s*start:\s*[-0-9.]+,\s*bitrate:\s*(\d+)\s*kb/s", stderr):
        duration = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
        container_bitrate = int(match.group(4))
    line = next((line.strip() for line in stderr.splitlines() if "Stream #0:" in line and "Audio:" in line), "")
    codec = ""; rate = None; layout = ""; channels = None
    if line:
        parts = [item.strip() for item in line.split("Audio:", 1)[1].split(",")]
        codec = parts[0].split()[0]
        if len(parts) > 1 and (match := re.search(r"(\d+)\s*Hz", parts[1])):
            rate = int(match.group(1))
        if len(parts) > 2:
            layout = parts[2]
            channels = 1 if layout == "mono" else 2 if layout == "stereo" else None
            if channels is None and (match := re.search(r"(\d+)\s*channels?", layout)):
                channels = int(match.group(1))
    return {"codec": codec, "sample_rate_hz": rate, "channels": channels, "channel_layout": layout,
            "ffmpeg_reported_duration_seconds": duration, "container_bitrate_kbps": container_bitrate}


def decode_audio(path: Path, ffmpeg: str) -> dict:
    command = [ffmpeg, "-hide_banner", "-nostdin", "-v", "info", "-i", str(path), "-map", "0:a:0",
               "-vn", "-sn", "-dn", "-t", "600", "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    timed_out = False
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900, creationflags=flags)
        raw = completed.stdout; stderr = completed.stderr.decode("utf-8", errors="replace"); code = completed.returncode
    except subprocess.TimeoutExpired as error:
        raw = error.stdout or b""; stderr = (error.stderr or b"").decode("utf-8", errors="replace"); code = 92; timed_out = True
    if len(raw) % 2:
        raw = raw[:-1]; code = code or 91
    samples = np.frombuffer(raw, dtype="<i2")
    absolute = np.abs(samples.astype(np.int32))
    silent = int(np.count_nonzero(absolute < SILENCE_THRESHOLD * 32768))
    clipped = int(np.count_nonzero(absolute >= CLIPPING_THRESHOLD * 32768))
    widened = samples.astype(np.int64)
    sum_squares = int(np.sum(widened * widened, dtype=np.int64)) if samples.size else 0
    result = parse_ffmpeg(stderr)
    result.update({
        "decode_status": "OK" if code == 0 and samples.size else "ERROR",
        "ffmpeg_return_code": code,
        "ffmpeg_wall_timeout": timed_out,
        "decoded_pcm_sha256": hashlib.sha256(raw).hexdigest() if samples.size else "",
        "decoded_sample_count": int(samples.size),
        "silent_sample_count": silent,
        "clipped_sample_count": clipped,
        "peak_abs_normalized": float(absolute.max() / 32768) if samples.size else 0.0,
        "rms_normalized": math.sqrt(sum_squares / samples.size) / 32768 if samples.size else 0.0,
        "silent_sample_fraction_lt_1e_4": silent / samples.size if samples.size else None,
        "clipped_sample_fraction_ge_0_999": clipped / samples.size if samples.size else None,
        "ffmpeg_error_excerpt": " | ".join(line.strip() for line in stderr.splitlines() if "error" in line.lower() or "invalid" in line.lower())[:1000],
    })
    result["decoded_duration_seconds"] = (
        samples.size / (result["sample_rate_hz"] * result["channels"])
        if samples.size and result["sample_rate_hz"] and result["channels"] else None
    )
    result["decode_hit_600_second_cap"] = bool(result["decoded_duration_seconds"] and result["decoded_duration_seconds"] >= 599.9)
    return result


def audit_file(path: Path, dataset_root: Path, manifest_rows: list[dict], ref_maps, ffmpeg: str) -> dict:
    relative = path.relative_to(dataset_root).as_posix()
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_BYTES): sha256.update(block)
    parts = relative.split("/")
    derived_type = parts[0] if len(parts) > 0 else ""
    derived_provider = parts[1] if len(parts) > 1 else ""
    if len(manifest_rows) == 1:
        manifest = dict(manifest_rows[0]); status = "UNIQUE_MATCH"
        reference = match_reference(manifest["original_audio"], *ref_maps)
    elif len(manifest_rows) > 1:
        manifest = {key: "" for key in ("original_audio", "generator", "type", "genre", "description", "duration", "manifest_row_number")}
        status = "AMBIGUOUS_DUPLICATE_PATH"
        reference = match_reference(" || ".join(sorted({row["original_audio"] for row in manifest_rows})), *ref_maps)
    else:
        manifest = {key: "" for key in ("original_audio", "generator", "type", "genre", "description", "duration", "manifest_row_number")}
        status = "UNREFERENCED_EXTRA"
        reference = match_reference(relative, *ref_maps)
    rights = provider_rights(derived_provider)
    result = {
        "relative_path": relative, "file_bytes": path.stat().st_size, "extension": path.suffix.lower(),
        "file_sha256": sha256.hexdigest(), "derived_type": derived_type, "derived_provider": derived_provider,
        "manifest_status": status, "manifest_row_count_for_path": len(manifest_rows),
        "manifest_candidate_rows": ";".join(str(row["manifest_row_number"]) for row in manifest_rows),
        "manifest_candidate_original_audio": " || ".join(row["original_audio"] for row in manifest_rows),
        "original_audio": manifest["original_audio"], "generator": manifest["generator"], "type": manifest["type"],
        "genre": manifest["genre"], "description": manifest["description"],
        "manifest_duration_seconds": float(manifest["duration"]) if manifest["duration"] else None,
        "manifest_row_number": manifest["manifest_row_number"],
    }
    result.update(reference); result.update(rights); result.update(inspect_signature(path)); result.update(decode_audio(path, ffmpeg))
    result["path_provider_matches_manifest"] = status == "UNIQUE_MATCH" and derived_provider == result["generator"]
    result["path_type_matches_manifest"] = status == "UNIQUE_MATCH" and derived_type == result["type"]
    if result["manifest_duration_seconds"] is not None and result["decoded_duration_seconds"] is not None:
        result["duration_delta_manifest_minus_decoded"] = result["manifest_duration_seconds"] - result["decoded_duration_seconds"]
        result["abs_duration_delta_manifest_decoded"] = abs(result["duration_delta_manifest_minus_decoded"])
    else:
        result["duration_delta_manifest_minus_decoded"] = None; result["abs_duration_delta_manifest_decoded"] = None
    issues = []
    if status != "UNIQUE_MATCH": issues.append(status.lower())
    if result["signature_status"] not in {"MP3_FRAMES_VALIDATED", "WAVE_VALIDATED"}: issues.append("signature_invalid")
    if result["decode_status"] != "OK": issues.append("decode_error")
    if status == "UNIQUE_MATCH" and not result["path_provider_matches_manifest"]: issues.append("provider_path_mismatch")
    if status == "UNIQUE_MATCH" and not result["path_type_matches_manifest"]: issues.append("type_path_mismatch")
    result["issues"] = ";".join(issues)
    return result


def quantiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size: return {"count": 0}
    return {"count": int(arr.size), "min": float(arr.min()), "p01": float(np.quantile(arr,.01)),
            "p05": float(np.quantile(arr,.05)), "p25": float(np.quantile(arr,.25)), "p50": float(np.quantile(arr,.5)),
            "p75": float(np.quantile(arr,.75)), "p95": float(np.quantile(arr,.95)), "p99": float(np.quantile(arr,.99)),
            "max": float(arr.max()), "mean": float(arr.mean()), "std_population": float(arr.std())}


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    args = parse_args(); root = args.root.resolve(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    archive = root / "Echoes.zip"; readme = root / "README.md"; dataset_root = root / "extracted" / "Echoes"
    manifest_path = dataset_root / "dataset_manifest.csv"; ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg: raise RuntimeError("ffmpeg not found")
    before = {"archive": file_state(archive), "readme": file_state(readme), "manifest": file_state(manifest_path)}
    archive_sha256 = hash_file(archive)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist(); member_count = len(members); file_members = [x for x in members if not x.is_dir()]
        unsafe = sum(not safe_member(x.filename) for x in members); uncompressed = sum(x.file_size for x in file_members)
        license_members = [x.filename for x in file_members if PurePosixPath(x.filename).name.lower().startswith("license")]
        crc_first_bad_member = bundle.testzip()
    manifest_rows = load_manifest(manifest_path); rows_by_path = defaultdict(list)
    for row in manifest_rows: rows_by_path[row["path_in_dataset"]].append(row)
    curated_ids = load_curated_ids(args.fma_metadata_root.resolve() / "tracks.csv")
    ref_maps = load_fma_references(args.fma_metadata_root.resolve() / "raw_tracks.csv", curated_ids, args.fma_allowlist.resolve())
    audio_paths = sorted([p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in {".mp3", ".wav"}], key=lambda p:p.as_posix())
    inventory = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_file, p, dataset_root, rows_by_path.get(p.relative_to(dataset_root).as_posix(), []), ref_maps, ffmpeg):p for p in audio_paths}
        for count, future in enumerate(as_completed(futures), start=1):
            inventory.append(future.result())
            if count % 200 == 0: print(f"audited {count}/{len(audio_paths)} Echoes audio files", flush=True)
    inventory.sort(key=lambda row: row["relative_path"])
    after = {"archive": file_state(archive), "readme": file_state(readme), "manifest": file_state(manifest_path)}

    file_groups = defaultdict(list); pcm_groups = defaultdict(list)
    for row in inventory:
        file_groups[row["file_sha256"]].append(row["relative_path"])
        if row["decoded_pcm_sha256"]: pcm_groups[row["decoded_pcm_sha256"]].append(row["relative_path"])
    file_dups = {k:v for k,v in file_groups.items() if len(v)>1}; pcm_dups = {k:v for k,v in pcm_groups.items() if len(v)>1}
    present_paths = {row["relative_path"] for row in inventory}; manifest_paths = set(rows_by_path)
    unique_rows = [row for row in inventory if row["manifest_status"] == "UNIQUE_MATCH"]
    durations = [float(row["decoded_duration_seconds"]) for row in inventory if row["decoded_duration_seconds"] is not None]
    silence = [float(row["silent_sample_fraction_lt_1e_4"]) for row in inventory if row["silent_sample_fraction_lt_1e_4"] is not None]
    clipping = [float(row["clipped_sample_fraction_ge_0_999"]) for row in inventory if row["clipped_sample_fraction_ge_0_999"] is not None]
    total_samples = sum(int(row["decoded_sample_count"]) for row in inventory); total_silent = sum(int(row["silent_sample_count"]) for row in inventory); total_clipped = sum(int(row["clipped_sample_count"]) for row in inventory)
    duplicate_manifest_paths = {path:len(rows) for path,rows in rows_by_path.items() if len(rows)>1}
    readme_text = readme.read_text(encoding="utf-8")
    audit = {
        "data_readiness": "BLOCKED",
        "block_reason": "Echoes contains generated audio only; provider generation-tier/output rights and bona-fide reference audio joins are not proven for automatic training use.",
        "method": {"scope":"all 4488 audio files; no sampling", "workers":args.workers, "decoder":subprocess.run([ffmpeg,"-version"],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0).stdout.splitlines()[0], "decoded_pcm_format":"s16le interleaved", "silence_definition":"abs(sample/32768)<1e-4", "clipping_definition":"abs(sample/32768)>=0.999", "max_decode_seconds":600},
        "source": {"root":str(root), "revision":root.name, "archive_bytes":archive.stat().st_size, "archive_sha256":archive_sha256, "archive_sha256_expected":EXPECTED_SHA256, "archive_sha256_matches_expected":archive_sha256==EXPECTED_SHA256, "zip_member_count":member_count, "zip_file_count":len(file_members), "zip_uncompressed_bytes":uncompressed, "zip_unsafe_member_count":unsafe, "zip_crc_first_bad_member":crc_first_bad_member, "zip_crc_all_ok":crc_first_bad_member is None, "zip_license_members":license_members, "extracted_file_count":sum(p.is_file() for p in dataset_root.rglob('*')), "extracted_bytes":sum(p.stat().st_size for p in dataset_root.rglob('*') if p.is_file()), "source_state_unchanged":before==after, "readme_front_matter_cc_by_sa_4_0":"license: cc-by-sa-4.0" in readme_text, "separate_license_file_present":bool(license_members)},
        "manifest": {"row_count":len(manifest_rows), "unique_path_count":len(manifest_paths), "duplicate_path_count":len(duplicate_manifest_paths), "duplicate_paths":duplicate_manifest_paths, "manifest_missing_audio_count":len(manifest_paths-present_paths), "audio_unreferenced_count":len(present_paths-manifest_paths), "audio_unreferenced_paths":sorted(present_paths-manifest_paths), "unique_match_audio_count":sum(r['manifest_status']=='UNIQUE_MATCH' for r in inventory), "ambiguous_path_audio_count":sum(r['manifest_status']=='AMBIGUOUS_DUPLICATE_PATH' for r in inventory), "provider_row_counts":dict(Counter(r['generator'] for r in manifest_rows)), "type_row_counts":dict(Counter(r['type'] for r in manifest_rows)), "genre_row_counts":dict(Counter(r['genre'] for r in manifest_rows)), "unique_original_audio_count":len({r['original_audio'] for r in manifest_rows}), "path_provider_mismatch_count":sum(r['manifest_status']=='UNIQUE_MATCH' and not r['path_provider_matches_manifest'] for r in inventory), "path_type_mismatch_count":sum(r['manifest_status']=='UNIQUE_MATCH' and not r['path_type_matches_manifest'] for r in inventory)},
        "audio": {"count":len(inventory), "extension_counts":dict(Counter(r['extension'] for r in inventory)), "provider_file_counts":dict(Counter(r['derived_provider'] for r in inventory)), "type_file_counts":dict(Counter(r['derived_type'] for r in inventory)), "decode_ok_count":sum(r['decode_status']=='OK' for r in inventory), "decode_error_count":sum(r['decode_status']!='OK' for r in inventory), "decode_error_paths":[r['relative_path'] for r in inventory if r['decode_status']!='OK'], "signature_ok_count":sum(r['signature_status'] in {'MP3_FRAMES_VALIDATED','WAVE_VALIDATED'} for r in inventory), "signature_bad_count":sum(r['signature_status'] not in {'MP3_FRAMES_VALIDATED','WAVE_VALIDATED'} for r in inventory), "codec_counts":dict(Counter(r['codec'] for r in inventory)), "sample_rate_counts":dict(Counter(str(r['sample_rate_hz']) for r in inventory)), "channel_counts":dict(Counter(str(r['channels']) for r in inventory)), "duration_distribution":quantiles(durations), "total_duration_hours":sum(durations)/3600, "manifest_abs_duration_delta_distribution":quantiles([float(r['abs_duration_delta_manifest_decoded']) for r in unique_rows if r['abs_duration_delta_manifest_decoded'] is not None]), "manifest_duration_delta_gt_0_1_count":sum(r['abs_duration_delta_manifest_decoded'] is not None and r['abs_duration_delta_manifest_decoded']>.1 for r in unique_rows), "total_samples":total_samples, "overall_silent_fraction":total_silent/total_samples, "silence_distribution":quantiles(silence), "silence_ge_50pct_count":sum(x>=.5 for x in silence), "overall_clipped_fraction":total_clipped/total_samples, "clipping_distribution":quantiles(clipping), "clipping_ge_1pct_count":sum(x>=.01 for x in clipping), "file_duplicate_group_count":len(file_dups), "file_duplicate_affected_count":sum(len(v) for v in file_dups.values()), "pcm_duplicate_group_count":len(pcm_dups), "pcm_duplicate_affected_count":sum(len(v) for v in pcm_dups.values())},
        "fma_reference_mapping": {"manifest_unique_original_audio":len({r['original_audio'] for r in manifest_rows}), "exact_unique_rows":sum(r['reference_match_status']=='EXACT_UNIQUE' for r in unique_rows), "normalized_unique_rows":sum(r['reference_match_status']=='NORMALIZED_UNIQUE' for r in unique_rows), "ambiguous_rows":sum('AMBIGUOUS' in r['reference_match_status'] for r in unique_rows), "unmatched_rows":sum(r['reference_match_status']=='UNMATCHED' for r in unique_rows), "unique_mapped_fma_ids":len({r['reference_fma_track_id'] for r in unique_rows if r['reference_fma_track_id']!=''}), "mapped_strict_allowlist_rows":sum(r['reference_strict_allowlist'] is True for r in unique_rows), "semantic_pair_group_count":len({r['semantic_pair_group'] for r in unique_rows})},
        "license_gate": {"dataset_surface_license":"README front matter CC BY-SA 4.0; no LICENSE inside ZIP", "all_training_eligible":False, "open_model_providers":sorted(OPEN_MODEL_PROVIDERS), "open_model_bucket_is_requested_review_policy_not_paper_commercial_flag":True, "commercial_providers":sorted(COMMERCIAL_PROVIDERS), "open_model_file_count":sum(r['derived_provider'] in OPEN_MODEL_PROVIDERS for r in inventory), "commercial_file_count":sum(r['derived_provider'] in COMMERCIAL_PROVIDERS for r in inventory), "conditional_hold_count":sum(r['conservative_training_eligibility']=='CONDITIONAL_HOLD' for r in inventory), "hold_count":sum(r['conservative_training_eligibility']=='HOLD' for r in inventory), "current_official_evidence":{"echoes_paper":"Table II claims provider model families/versions and labels nine systems commercial; this does not identify an exact checkpoint/API revision or prove output rights", "suno":"current terms: paid Pro/Premier output assignment; free/Basic personal non-commercial with attribution; Output may not be used to compete with Suno", "elevenlabs":"current terms: user retains output rights; free use non-commercial, paid may be commercial", "musicgen":"official model card: code MIT, weights CC-BY-NC 4.0"}, "corpus_generation_tier_documented":False},
    }
    fields = ["relative_path","file_bytes","extension","file_sha256","derived_type","derived_provider","manifest_status","manifest_row_count_for_path","manifest_candidate_rows","manifest_candidate_original_audio","manifest_row_number","original_audio","generator","type","genre","description","manifest_duration_seconds","path_provider_matches_manifest","path_type_matches_manifest","reference_match_status","reference_fma_track_id","reference_fma_candidate_ids","reference_artist_id","reference_album_id","reference_strict_allowlist","reference_license_category","semantic_pair_group","provider_license_tier","evidence_status","conservative_training_eligibility","training_eligible","signature_status","id3v2_signature","decode_status","ffmpeg_return_code","ffmpeg_wall_timeout","decode_hit_600_second_cap","codec","sample_rate_hz","channels","channel_layout","container_bitrate_kbps","ffmpeg_reported_duration_seconds","decoded_duration_seconds","duration_delta_manifest_minus_decoded","abs_duration_delta_manifest_decoded","decoded_pcm_sha256","decoded_sample_count","peak_abs_normalized","rms_normalized","silent_sample_count","silent_sample_fraction_lt_1e_4","clipped_sample_count","clipped_sample_fraction_ge_0_999","ffmpeg_error_excerpt","issues"]
    csv_write(out/"echoes-audio-inventory.csv",inventory,fields)
    curated = [r for r in inventory if r['manifest_status']=='UNIQUE_MATCH']
    csv_write(out/"echoes-curated-manifest.csv",curated,fields)
    dups=[]
    for kind,groups in (("file_sha256",file_dups),("decoded_pcm_sha256",pcm_dups)):
        for digest,paths in sorted(groups.items()): dups.append({"duplicate_type":kind,"sha256":digest,"member_count":len(paths),"paths":";".join(sorted(paths))})
    csv_write(out/"echoes-duplicate-groups.csv",dups,["duplicate_type","sha256","member_count","paths"])
    with (out/"echoes-audit-run.json").open("w",encoding="utf-8") as stream: json.dump(audit,stream,indent=2,ensure_ascii=False);stream.write("\n")
    print(json.dumps({"audio":len(inventory),"decode_errors":audit['audio']['decode_error_count'],"unique_manifest":audit['manifest']['unique_match_audio_count'],"ambiguous":audit['manifest']['ambiguous_path_audio_count'],"extras":audit['manifest']['audio_unreferenced_count'],"mapped_fma_ids":audit['fma_reference_mapping']['unique_mapped_fma_ids'],"file_dups":audit['audio']['file_duplicate_group_count'],"pcm_dups":audit['audio']['pcm_duplicate_group_count']},indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
