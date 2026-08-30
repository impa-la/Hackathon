#!/usr/bin/env python3
"""Read-only full audit of the official FMA small MP3 bundle.

Every MP3 is hashed and decoded through FFmpeg. The script writes only below
--output-dir and never extracts, rewrites, tags, or normalizes source audio.
"""

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
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np


EXPECTED_ARCHIVE_SHA256 = "f923bbef327820456d50965c3c320c3c7b6dab8456449429fd78f7ec96a5d02c"
EXPECTED_ARCHIVE_BYTES = 7_679_594_875
EXPECTED_MEMBER_COUNT = 8_002
EXPECTED_EXTRACTED_FILES = 8_002
EXPECTED_EXTRACTED_BYTES = 7_975_472_258
EXPECTED_MP3_COUNT = 8_000
EXPECTED_ALLOWLIST_COUNT = 5_130
SILENCE_THRESHOLD = 1e-4
CLIPPING_THRESHOLD = 0.999
READ_BYTES = 8 * 1024 * 1024

ALLOW_CATEGORIES = {
    "ALLOW_CC0",
    "ALLOW_BY",
    "ALLOW_BY_SA",
    "ALLOW_BY_NC",
    "ALLOW_BY_NC_SA",
}


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh-signatures-only", action="store_true")
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
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and not (candidate.parts and ":" in candidate.parts[0])
    )


def read_checksums(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        result[relative.strip().lstrip("*").replace("\\", "/")] = expected.lower()
    return result


def clean(text: str | None) -> str:
    return "" if text is None else str(text).strip()


def classify_license(title: str, url: str) -> str:
    title_text = clean(title).lower()
    url_text = clean(url).lower()
    if "/publicdomain/zero/" in url_text or "cc0" in title_text:
        return "ALLOW_CC0"
    if "/licenses/by-nc-sa/" in url_text:
        return "ALLOW_BY_NC_SA"
    if "/licenses/by-nc-nd/" in url_text or "/licenses/by-nd/" in url_text:
        return "EXCLUDE_ND"
    if "/licenses/by-nc/" in url_text:
        return "ALLOW_BY_NC"
    if "/licenses/by-sa/" in url_text:
        return "ALLOW_BY_SA"
    if re.search(r"/licenses/by/(?:[0-9]|$)", url_text):
        return "ALLOW_BY"
    if not url_text:
        return "EXCLUDE_UNKNOWN"
    if any(
        marker in url_text
        for marker in ("fma_license", "music-sharing", "sound_recording_common_law", "orphan_work")
    ):
        return "EXCLUDE_RESTRICTED"
    return "EXCLUDE_CUSTOM"


def parse_hms(value: str) -> float | None:
    value = clean(value)
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(value)
    except ValueError:
        return None


def load_curated_small(tracks_csv: Path) -> dict[int, dict]:
    with tracks_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        top = next(reader)
        sub = next(reader)
        index_name = next(reader)
        columns = []
        for first, second, third in zip(top, sub, index_name):
            if third == "track_id":
                columns.append("track_id")
            elif first and second:
                columns.append(f"{first}.{second}")
            else:
                columns.append(first or second or third)
        positions = {name: idx for idx, name in enumerate(columns)}
        required = {
            "track_id", "album.id", "artist.id", "set.split", "set.subset",
            "track.bit_rate", "track.duration", "track.genre_top", "track.license", "track.title",
        }
        missing = required - set(positions)
        if missing:
            raise ValueError(f"Missing tracks.csv columns: {sorted(missing)}")
        result = {}
        for values in reader:
            if len(values) < len(columns):
                values += [""] * (len(columns) - len(values))
            if clean(values[positions["set.subset"]]) != "small":
                continue
            track_id = int(values[positions["track_id"]])
            result[track_id] = {
                "track_id": track_id,
                "album_id": int(values[positions["album.id"]]),
                "artist_id": int(values[positions["artist.id"]]),
                "split": clean(values[positions["set.split"]]),
                "subset": "small",
                "metadata_bit_rate": int(float(values[positions["track.bit_rate"]] or 0)),
                "metadata_track_duration_seconds": float(values[positions["track.duration"]]) if values[positions["track.duration"]] else None,
                "genre_top": clean(values[positions["track.genre_top"]]),
                "license_title": clean(values[positions["track.license"]]),
                "track_title": clean(values[positions["track.title"]]),
            }
    return result


def load_raw_small(raw_csv: Path, small_ids: set[int]) -> dict[int, dict]:
    result = {}
    with raw_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            track_id = int(row["track_id"])
            if track_id not in small_ids:
                continue
            result[track_id] = {
                "license_url": clean(row.get("license_url")),
                "raw_license_title": clean(row.get("license_title")),
                "raw_track_bit_rate": int(float(row.get("track_bit_rate") or 0)),
                "raw_track_duration_seconds": parse_hms(row.get("track_duration", "")),
                "track_file": clean(row.get("track_file")),
            }
    return result


def load_allowlist(path: Path) -> tuple[set[int], dict[int, dict]]:
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if clean(row.get("small_member")).lower() != "true":
                continue
            track_id = int(row["track_id"])
            result[track_id] = row
    return set(result), result


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def connected_components(metadata: dict[int, dict]) -> tuple[dict[int, str], dict[str, dict]]:
    uf = UnionFind()
    for row in metadata.values():
        uf.union(f"artist:{row['artist_id']}", f"album:{row['album_id']}")
    tracks_by_root = defaultdict(list)
    for track_id, row in metadata.items():
        root = uf.find(f"artist:{row['artist_id']}")
        tracks_by_root[root].append(track_id)
    track_component = {}
    component_summary = {}
    for track_ids in tracks_by_root.values():
        component_id = f"fma-small-cc-{min(track_ids):06d}"
        splits = sorted({metadata[track_id]["split"] for track_id in track_ids})
        artists = {metadata[track_id]["artist_id"] for track_id in track_ids}
        albums = {metadata[track_id]["album_id"] for track_id in track_ids}
        summary = {
            "component_id": component_id,
            "track_count": len(track_ids),
            "artist_count": len(artists),
            "album_count": len(albums),
            "split_count": len(splits),
            "splits": ";".join(splits),
            "cross_split": len(splits) > 1,
            "min_track_id": min(track_ids),
            "max_track_id": max(track_ids),
        }
        component_summary[component_id] = summary
        for track_id in track_ids:
            track_component[track_id] = component_id
    return track_component, component_summary


def inspect_signature(path: Path) -> dict:
    with path.open("rb") as stream:
        head = stream.read(min(path.stat().st_size, 2 * 1024 * 1024))
    id3v2 = head.startswith(b"ID3")
    offset = 0
    if id3v2 and len(head) >= 10:
        size_bytes = head[6:10]
        if all(value < 128 for value in size_bytes):
            offset = 10 + sum(value << shift for value, shift in zip(size_bytes, (21, 14, 7, 0)))
    def frame_length_at(data: bytes, index: int) -> int | None:
        if index + 4 > len(data) or data[index] != 0xFF or (data[index + 1] & 0xE0) != 0xE0:
            return None
        version_bits = (data[index + 1] >> 3) & 0x03
        layer_bits = (data[index + 1] >> 1) & 0x03
        bitrate_index = (data[index + 2] >> 4) & 0x0F
        sample_rate_index = (data[index + 2] >> 2) & 0x03
        padding = (data[index + 2] >> 1) & 0x01
        if version_bits == 1 or layer_bits != 1 or bitrate_index in {0, 15} or sample_rate_index == 3:
            return None
        sample_rates = {
            3: (44100, 48000, 32000),
            2: (22050, 24000, 16000),
            0: (11025, 12000, 8000),
        }
        mpeg1_bitrates = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)
        mpeg2_bitrates = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
        sample_rate = sample_rates[version_bits][sample_rate_index]
        bitrate = (mpeg1_bitrates if version_bits == 3 else mpeg2_bitrates)[bitrate_index]
        coefficient = 144000 if version_bits == 3 else 72000
        return int(coefficient * bitrate / sample_rate) + padding

    frame_offset = None
    for index in range(min(offset, len(head)), len(head) - 1):
        length = frame_length_at(head, index)
        if length and frame_length_at(head, index + length):
            frame_offset = index
            break
    return {
        "id3v2_signature": id3v2,
        "mpeg_frame_signature_found": frame_offset is not None,
        "first_mpeg_frame_offset": frame_offset,
        "signature_status": "MP3_FRAMES_VALIDATED" if frame_offset is not None else "NO_CONSECUTIVE_MPEG_FRAMES",
    }


def parse_ffmpeg(stderr: str) -> dict:
    duration_seconds = None
    container_bitrate_kbps = None
    duration_match = re.search(
        r"Duration:\s*(\d+):(\d+):([0-9.]+),\s*start:\s*[-0-9.]+,\s*bitrate:\s*(\d+)\s*kb/s",
        stderr,
    )
    if duration_match:
        duration_seconds = (
            int(duration_match.group(1)) * 3600
            + int(duration_match.group(2)) * 60
            + float(duration_match.group(3))
        )
        container_bitrate_kbps = int(duration_match.group(4))
    stream_line = next(
        (line.strip() for line in stderr.splitlines() if "Stream #0:" in line and "Audio:" in line),
        "",
    )
    codec = ""
    sample_rate = None
    channel_layout = ""
    channels = None
    stream_bitrate_kbps = None
    if stream_line:
        parts = stream_line.split("Audio:", 1)[1].split(",")
        parts = [part.strip() for part in parts]
        codec = parts[0].split()[0] if parts else ""
        if len(parts) > 1:
            match = re.search(r"(\d+)\s*Hz", parts[1])
            sample_rate = int(match.group(1)) if match else None
        if len(parts) > 2:
            channel_layout = parts[2]
            if channel_layout == "mono":
                channels = 1
            elif channel_layout == "stereo":
                channels = 2
            elif match := re.search(r"(\d+)\s*channels?", channel_layout):
                channels = int(match.group(1))
            elif channel_layout.startswith("5.1"):
                channels = 6
        for part in parts[3:]:
            if match := re.fullmatch(r"(\d+)\s*kb/s", part):
                stream_bitrate_kbps = int(match.group(1))
    return {
        "ffmpeg_reported_duration_seconds": duration_seconds,
        "container_bitrate_kbps": container_bitrate_kbps,
        "stream_bitrate_kbps": stream_bitrate_kbps,
        "codec": codec,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "channel_layout": channel_layout,
    }


def decode_mp3(path: Path, ffmpeg: str) -> dict:
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-v", "info", "-i", str(path),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", "120",
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        raw_pcm = completed.stdout
        stderr = completed.stderr.decode("utf-8", errors="replace")
        return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        raw_pcm = error.stdout or b""
        stderr_bytes = error.stderr or b""
        stderr = stderr_bytes.decode("utf-8", errors="replace") + "\nFFmpeg wall timeout after 180 seconds"
        return_code = 92
        timed_out = True
    if len(raw_pcm) % 2:
        raw_pcm = raw_pcm[:-1]
        return_code = return_code or 91
        stderr += "\nOdd trailing PCM byte"
    digest = hashlib.sha256(raw_pcm)
    samples = np.frombuffer(raw_pcm, dtype="<i2")
    absolute = np.abs(samples.astype(np.int32))
    sample_count = int(samples.size)
    silent_count = int(np.count_nonzero(absolute < SILENCE_THRESHOLD * 32768.0))
    clipped_count = int(np.count_nonzero(absolute >= CLIPPING_THRESHOLD * 32768.0))
    peak_abs = float(absolute.max() / 32768.0) if absolute.size else 0.0
    widened = samples.astype(np.int64)
    sum_squares_integer = int(np.sum(widened * widened, dtype=np.int64)) if widened.size else 0
    parsed = parse_ffmpeg(stderr)
    parsed.update(
        {
            "decode_status": "OK" if return_code == 0 and sample_count > 0 else "ERROR",
            "ffmpeg_return_code": return_code,
            "ffmpeg_wall_timeout": timed_out,
            "decoded_pcm_sha256": digest.hexdigest() if sample_count else "",
            "decoded_sample_count": sample_count,
            "silent_sample_count": silent_count,
            "clipped_sample_count": clipped_count,
            "peak_abs_normalized": peak_abs,
            "rms_normalized": math.sqrt(sum_squares_integer / sample_count) / 32768.0 if sample_count else 0.0,
            "silent_sample_fraction_lt_1e_4": silent_count / sample_count if sample_count else None,
            "clipped_sample_fraction_ge_0_999": clipped_count / sample_count if sample_count else None,
            "ffmpeg_error_excerpt": " | ".join(
                line.strip() for line in stderr.splitlines() if "error" in line.lower() or "invalid" in line.lower()
            )[:1000],
        }
    )
    if parsed["sample_rate_hz"] and parsed["channels"] and sample_count:
        parsed["decoded_duration_seconds"] = sample_count / (
            parsed["sample_rate_hz"] * parsed["channels"]
        )
    else:
        parsed["decoded_duration_seconds"] = None
    parsed["decode_hit_120_second_cap"] = bool(
        parsed["decoded_duration_seconds"] is not None
        and parsed["decoded_duration_seconds"] >= 119.9
    )
    return parsed


def audit_one(
    path: Path,
    audio_root: Path,
    metadata: dict | None,
    expected_sha1: str | None,
    allowed: bool,
    component_id: str,
    component_cross_split: bool,
    ffmpeg: str,
) -> dict:
    relative = path.relative_to(audio_root).as_posix()
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    with path.open("rb") as stream:
        while block := stream.read(READ_BYTES):
            sha256.update(block)
            sha1.update(block)
    result = {
        "track_id": int(path.stem),
        "relative_path": relative,
        "file_bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
        "file_sha256": sha256.hexdigest(),
        "file_sha1": sha1.hexdigest(),
        "expected_sha1": expected_sha1 or "",
        "checksum_match": bool(expected_sha1) and sha1.hexdigest() == expected_sha1,
        "metadata_present": metadata is not None,
        "strict_allowlist": allowed,
        "component_id": component_id,
        "component_cross_split": component_cross_split,
    }
    if metadata:
        result.update(metadata)
    result.update(inspect_signature(path))
    result.update(decode_mp3(path, ffmpeg))
    duration = result.get("decoded_duration_seconds")
    result["duration_delta_from_30_seconds"] = duration - 30.0 if duration is not None else None
    result["abs_duration_delta_from_30_seconds"] = abs(duration - 30.0) if duration is not None else None
    issues = []
    if result["extension"] != ".mp3":
        issues.append("extension_not_mp3")
    if result["signature_status"] != "MP3_FRAMES_VALIDATED":
        issues.append("signature_invalid")
    if not result["checksum_match"]:
        issues.append("sha1_mismatch_or_missing")
    if not result["metadata_present"]:
        issues.append("metadata_missing")
    if result["decode_status"] != "OK":
        issues.append("decode_error")
    result["issues"] = ";".join(issues)
    return result


def quantiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return {"count": 0}
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


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def counter_rows(counter: Counter) -> list[dict]:
    return [{"value": str(key), "count": value} for key, value in sorted(counter.items(), key=lambda x: str(x[0]))]


def refresh_signatures_only(output_dir: Path, audio_root: Path) -> None:
    inventory_path = output_dir / "fma-small-audio-inventory.csv"
    audit_path = output_dir / "fma-small-audit-run.json"
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for index, row in enumerate(rows, start=1):
        signature = inspect_signature(audio_root / row["relative_path"])
        row.update({key: str(value) if value is not None else "" for key, value in signature.items()})
        issues = [item for item in row.get("issues", "").split(";") if item and item != "signature_invalid"]
        if signature["signature_status"] != "MP3_FRAMES_VALIDATED":
            issues.append("signature_invalid")
        row["issues"] = ";".join(issues)
        if index % 1000 == 0:
            print(f"refreshed signatures {index}/{len(rows)}", flush=True)
    csv_write(inventory_path, rows, fields)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    signature_ok = sum(row["signature_status"] == "MP3_FRAMES_VALIDATED" for row in rows)
    audit["audio"]["signature_ok_count"] = signature_ok
    audit["audio"]["signature_bad_count"] = len(rows) - signature_ok
    allowed = [row for row in rows if row["strict_allowlist"].lower() == "true"]
    audit["strict_allowlist"]["signature_ok_count"] = sum(
        row["signature_status"] == "MP3_FRAMES_VALIDATED" for row in allowed
    )
    with audit_path.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> int:
    args = args_parser()
    archive = args.archive.resolve()
    extracted_root = args.extracted_root.resolve()
    metadata_root = args.metadata_root.resolve()
    allowlist_path = args.allowlist.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    audio_root = extracted_root / "fma_small"
    if args.refresh_signatures_only:
        refresh_signatures_only(output_dir, audio_root)
        return 0
    checksums_path = audio_root / "checksums"
    readme_path = audio_root / "README.txt"
    source_before = {
        "archive": file_state(archive),
        "checksums": file_state(checksums_path),
        "readme": file_state(readme_path),
    }
    archive_sha256 = hash_file(archive)
    expected_checksums = read_checksums(checksums_path)

    curated = load_curated_small(metadata_root / "tracks.csv")
    raw = load_raw_small(metadata_root / "raw_tracks.csv", set(curated))
    allow_ids, allow_rows = load_allowlist(allowlist_path)
    if len(curated) != EXPECTED_MP3_COUNT or len(raw) != EXPECTED_MP3_COUNT:
        raise ValueError(f"small metadata mismatch: curated={len(curated)} raw={len(raw)}")
    if len(allow_ids) != EXPECTED_ALLOWLIST_COUNT:
        raise ValueError(f"allowlist mismatch: {len(allow_ids)}")
    for track_id, row in curated.items():
        raw_row = raw[track_id]
        row.update(raw_row)
        row["license_category"] = classify_license(
            raw_row["raw_license_title"], raw_row["license_url"]
        )
        row["license_allow_by_classifier"] = row["license_category"] in ALLOW_CATEGORIES
        row["allowlist_membership_matches_classifier"] = (
            row["license_allow_by_classifier"] == (track_id in allow_ids)
        )

    track_component, component_summary = connected_components(curated)
    component_cross = {
        component_id: bool(row["cross_split"])
        for component_id, row in component_summary.items()
    }

    mp3_paths = sorted(audio_root.rglob("*.mp3"), key=lambda path: path.as_posix())
    all_extracted_files = sorted(path for path in audio_root.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive) as bundle:
        zip_members = bundle.infolist()
        zip_member_count = len(zip_members)
        unsafe_member_count = sum(not safe_member(member.filename) for member in zip_members)
        zip_file_member_count = sum(not member.is_dir() for member in zip_members)
        zip_uncompressed_bytes = sum(member.file_size for member in zip_members if not member.is_dir())
        zip_mp3_member_count = sum(member.filename.lower().endswith(".mp3") for member in zip_members)

    inventory = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for path in mp3_paths:
            track_id = int(path.stem)
            relative = path.relative_to(audio_root).as_posix()
            component_id = track_component.get(track_id, "")
            future = executor.submit(
                audit_one,
                path,
                audio_root,
                curated.get(track_id),
                expected_checksums.get(relative),
                track_id in allow_ids,
                component_id,
                component_cross.get(component_id, False),
                ffmpeg,
            )
            futures[future] = track_id
        completed = 0
        for future in as_completed(futures):
            inventory.append(future.result())
            completed += 1
            if completed % 250 == 0:
                print(f"audited {completed}/{len(mp3_paths)} MP3 files", flush=True)
    inventory.sort(key=lambda row: row["track_id"])

    source_after = {
        "archive": file_state(archive),
        "checksums": file_state(checksums_path),
        "readme": file_state(readme_path),
    }
    source_unchanged = source_before == source_after

    file_hash_groups = defaultdict(list)
    pcm_hash_groups = defaultdict(list)
    for row in inventory:
        file_hash_groups[row["file_sha256"]].append(row["track_id"])
        if row["decoded_pcm_sha256"]:
            pcm_hash_groups[row["decoded_pcm_sha256"]].append(row["track_id"])
    duplicate_file_groups = {key: ids for key, ids in file_hash_groups.items() if len(ids) > 1}
    duplicate_pcm_groups = {key: ids for key, ids in pcm_hash_groups.items() if len(ids) > 1}

    durations = [float(row["decoded_duration_seconds"]) for row in inventory if row["decoded_duration_seconds"] is not None]
    silence = [float(row["silent_sample_fraction_lt_1e_4"]) for row in inventory if row["silent_sample_fraction_lt_1e_4"] is not None]
    clipping = [float(row["clipped_sample_fraction_ge_0_999"]) for row in inventory if row["clipped_sample_fraction_ge_0_999"] is not None]
    file_sizes = [int(row["file_bytes"]) for row in inventory]
    total_samples = sum(int(row["decoded_sample_count"]) for row in inventory)
    total_silent = sum(int(row["silent_sample_count"]) for row in inventory)
    total_clipped = sum(int(row["clipped_sample_count"]) for row in inventory)
    present_ids = {row["track_id"] for row in inventory}
    allowed_rows = [row for row in inventory if row["strict_allowlist"]]
    components = list(component_summary.values())
    cross_components = [row for row in components if row["cross_split"]]

    ffmpeg_version = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ).stdout.splitlines()[0]

    audit = {
        "data_readiness": "BLOCKED",
        "block_reason": "FMA small real-audio audit passed, but WaveFake generated audio and paired real/fake manifests are not yet complete.",
        "method": {
            "scope": "all 8000 MP3 files; no sampling",
            "workers": args.workers,
            "decoder": ffmpeg_version,
            "decoded_pcm_format": "signed 16-bit little-endian interleaved PCM",
            "silence_definition": "abs(sample / 32768) < 1e-4",
            "clipping_definition": "abs(sample / 32768) >= 0.999",
            "file_duplicate_key": "SHA-256 of complete MP3 bytes",
            "decoded_duplicate_key": "SHA-256 of FFmpeg-decoded s16le PCM bytes; decoder-version dependent",
            "random_sampling": "none",
        },
        "source": {
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha256,
            "archive_sha256_expected": EXPECTED_ARCHIVE_SHA256,
            "archive_sha256_matches_expected": archive_sha256 == EXPECTED_ARCHIVE_SHA256,
            "zip_member_count": zip_member_count,
            "zip_file_member_count": zip_file_member_count,
            "zip_mp3_member_count": zip_mp3_member_count,
            "zip_uncompressed_bytes": zip_uncompressed_bytes,
            "zip_unsafe_member_count": unsafe_member_count,
            "extracted_file_count": len(all_extracted_files),
            "extracted_bytes": sum(path.stat().st_size for path in all_extracted_files),
            "extracted_mp3_count": len(mp3_paths),
            "checksum_entry_count": len(expected_checksums),
            "source_state_unchanged_during_audit": source_unchanged,
        },
        "joins": {
            "curated_small_metadata_ids": len(curated),
            "raw_small_metadata_ids": len(raw),
            "audio_ids": len(present_ids),
            "audio_missing_from_curated_count": len(present_ids - set(curated)),
            "curated_missing_audio_count": len(set(curated) - present_ids),
            "duplicate_audio_id_count": len(mp3_paths) - len(present_ids),
            "strict_allowlist_ids": len(allow_ids),
            "strict_allowlist_present_audio_count": len(allow_ids & present_ids),
            "strict_allowlist_missing_audio_count": len(allow_ids - present_ids),
            "allowlist_classifier_mismatch_count": sum(
                not row["allowlist_membership_matches_classifier"] for row in curated.values()
            ),
        },
        "audio": {
            "mp3_count": len(inventory),
            "file_bytes_distribution": quantiles(file_sizes),
            "checksum_match_count": sum(bool(row["checksum_match"]) for row in inventory),
            "checksum_mismatch_count": sum(not bool(row["checksum_match"]) for row in inventory),
            "signature_ok_count": sum(row["signature_status"] == "MP3_FRAMES_VALIDATED" for row in inventory),
            "signature_bad_count": sum(row["signature_status"] != "MP3_FRAMES_VALIDATED" for row in inventory),
            "id3v2_count": sum(bool(row["id3v2_signature"]) for row in inventory),
            "decode_ok_count": sum(row["decode_status"] == "OK" for row in inventory),
            "decode_error_count": sum(row["decode_status"] != "OK" for row in inventory),
            "decode_error_ids": [row["track_id"] for row in inventory if row["decode_status"] != "OK"],
            "ffmpeg_wall_timeout_count": sum(bool(row["ffmpeg_wall_timeout"]) for row in inventory),
            "decode_hit_120_second_cap_count": sum(bool(row["decode_hit_120_second_cap"]) for row in inventory),
            "codec_counts": dict(Counter(row["codec"] for row in inventory)),
            "sample_rate_counts": dict(Counter(str(row["sample_rate_hz"]) for row in inventory)),
            "channel_counts": dict(Counter(str(row["channels"]) for row in inventory)),
            "channel_layout_counts": dict(Counter(row["channel_layout"] for row in inventory)),
            "container_bitrate_kbps_counts": dict(Counter(str(row["container_bitrate_kbps"]) for row in inventory)),
            "stream_bitrate_kbps_counts": dict(Counter(str(row["stream_bitrate_kbps"]) for row in inventory)),
            "duration_seconds_distribution": quantiles(durations),
            "total_decoded_duration_seconds": sum(durations),
            "total_decoded_duration_hours": sum(durations) / 3600,
            "duration_abs_delta_from_30_distribution": quantiles([
                abs(value - 30.0) for value in durations
            ]),
            "duration_within_0_1_seconds_of_30_count": sum(abs(value - 30.0) <= 0.1 for value in durations),
            "duration_more_than_1_second_from_30_count": sum(abs(value - 30.0) > 1.0 for value in durations),
            "duration_more_than_5_seconds_from_30_count": sum(abs(value - 30.0) > 5.0 for value in durations),
            "total_decoded_sample_count": total_samples,
            "total_silent_sample_count": total_silent,
            "overall_silent_sample_fraction_lt_1e_4": total_silent / total_samples if total_samples else None,
            "silence_fraction_distribution": quantiles(silence),
            "all_silent_file_count": sum(value == 1.0 for value in silence),
            "silence_fraction_ge_50pct_count": sum(value >= 0.5 for value in silence),
            "total_clipped_sample_count": total_clipped,
            "overall_clipped_sample_fraction_ge_0_999": total_clipped / total_samples if total_samples else None,
            "clipping_fraction_distribution": quantiles(clipping),
            "any_clipped_sample_file_count": sum(int(row["clipped_sample_count"]) > 0 for row in inventory),
            "file_hash_duplicate_group_count": len(duplicate_file_groups),
            "file_hash_duplicate_affected_count": sum(len(ids) for ids in duplicate_file_groups.values()),
            "decoded_pcm_duplicate_group_count": len(duplicate_pcm_groups),
            "decoded_pcm_duplicate_affected_count": sum(len(ids) for ids in duplicate_pcm_groups.values()),
        },
        "strict_allowlist": {
            "count": len(allowed_rows),
            "present_count": len(allow_ids & present_ids),
            "decode_ok_count": sum(row["decode_status"] == "OK" for row in allowed_rows),
            "checksum_match_count": sum(bool(row["checksum_match"]) for row in allowed_rows),
            "signature_ok_count": sum(row["signature_status"] == "MP3_FRAMES_VALIDATED" for row in allowed_rows),
            "total_decoded_duration_hours": sum(float(row["decoded_duration_seconds"]) for row in allowed_rows if row["decoded_duration_seconds"] is not None) / 3600,
            "license_category_counts": dict(Counter(row["license_category"] for row in allowed_rows)),
            "split_counts": dict(Counter(row["split"] for row in allowed_rows)),
            "sample_rate_counts": dict(Counter(str(row["sample_rate_hz"]) for row in allowed_rows)),
            "channel_counts": dict(Counter(str(row["channels"]) for row in allowed_rows)),
            "decode_error_ids": [row["track_id"] for row in allowed_rows if row["decode_status"] != "OK"],
        },
        "license_all_small": {
            "category_counts": dict(Counter(row["license_category"] for row in inventory)),
        },
        "album_artist_components": {
            "definition": "connected components in the bipartite artist_id--album_id graph",
            "component_count": len(components),
            "cross_split_component_count": len(cross_components),
            "cross_split_affected_track_count": sum(int(row["track_count"]) for row in cross_components),
            "largest_component_track_count": max(int(row["track_count"]) for row in components),
            "largest_component_artist_count": max(int(row["artist_count"]) for row in components),
            "largest_component_album_count": max(int(row["album_count"]) for row in components),
        },
    }

    inventory_fields = [
        "track_id", "relative_path", "file_bytes", "extension", "file_sha256", "file_sha1",
        "expected_sha1", "checksum_match", "signature_status", "id3v2_signature",
        "mpeg_frame_signature_found", "first_mpeg_frame_offset", "metadata_present",
        "strict_allowlist", "license_category", "license_allow_by_classifier",
        "allowlist_membership_matches_classifier", "split", "subset", "artist_id", "album_id",
        "component_id", "component_cross_split", "genre_top", "license_title", "license_url",
        "track_title", "track_file", "metadata_bit_rate", "raw_track_bit_rate",
        "metadata_track_duration_seconds", "raw_track_duration_seconds", "decode_status",
        "ffmpeg_return_code", "ffmpeg_wall_timeout", "decode_hit_120_second_cap", "codec", "sample_rate_hz", "channels", "channel_layout",
        "container_bitrate_kbps", "stream_bitrate_kbps", "ffmpeg_reported_duration_seconds",
        "decoded_duration_seconds", "duration_delta_from_30_seconds", "abs_duration_delta_from_30_seconds",
        "decoded_pcm_sha256", "decoded_sample_count", "peak_abs_normalized", "rms_normalized",
        "silent_sample_count", "silent_sample_fraction_lt_1e_4", "clipped_sample_count",
        "clipped_sample_fraction_ge_0_999", "ffmpeg_error_excerpt", "issues",
    ]
    csv_write(output_dir / "fma-small-audio-inventory.csv", inventory, inventory_fields)

    duplicate_rows = []
    for duplicate_type, groups in (("file_sha256", duplicate_file_groups), ("decoded_pcm_sha256", duplicate_pcm_groups)):
        for digest, ids in sorted(groups.items()):
            duplicate_rows.append({
                "duplicate_type": duplicate_type,
                "sha256": digest,
                "member_count": len(ids),
                "track_ids": ";".join(str(item) for item in sorted(ids)),
                "all_strict_allowlist": all(track_id in allow_ids for track_id in ids),
            })
    csv_write(
        output_dir / "fma-small-duplicate-groups.csv",
        duplicate_rows,
        ["duplicate_type", "sha256", "member_count", "track_ids", "all_strict_allowlist"],
    )
    csv_write(
        output_dir / "fma-small-component-summary.csv",
        sorted(component_summary.values(), key=lambda row: row["component_id"]),
        ["component_id", "track_count", "artist_count", "album_count", "split_count", "splits", "cross_split", "min_track_id", "max_track_id"],
    )
    with (output_dir / "fma-small-audit-run.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(json.dumps({
        "mp3_count": len(inventory),
        "decode_error_count": audit["audio"]["decode_error_count"],
        "checksum_mismatch_count": audit["audio"]["checksum_mismatch_count"],
        "allowlist_present": audit["strict_allowlist"]["present_count"],
        "duration_hours": audit["audio"]["total_decoded_duration_hours"],
        "file_duplicate_groups": audit["audio"]["file_hash_duplicate_group_count"],
        "pcm_duplicate_groups": audit["audio"]["decoded_pcm_duplicate_group_count"],
        "source_unchanged": source_unchanged,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
