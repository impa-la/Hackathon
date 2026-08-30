"""Read-only audit of the FMA metadata archive for DeepVoice data gating.

No network requests are made. The source ZIP and extracted files are opened
read-only. Outputs are written only below the caller-provided output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np
import pandas as pd


CHUNK = 8 * 1024 * 1024
ALLOW_CATEGORIES = {
    "ALLOW_CC0",
    "ALLOW_BY",
    "ALLOW_BY_SA",
    "ALLOW_BY_NC",
    "ALLOW_BY_NC_SA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name: str) -> bool:
    candidate = PurePosixPath(name.replace("\\", "/"))
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and not (candidate.parts and ":" in candidate.parts[0])
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_checksum_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        result[name.strip().lstrip("*")] = expected.lower()
    return result


class RestrictedFmaUnpickler(pickle.Unpickler):
    """Allow only the two NumPy constructors used by FMA not_found.pickle."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("numpy.core.multiarray", "scalar"):
            return np._core.multiarray.scalar
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")


def load_not_found(path: Path) -> dict[str, list[Any]]:
    with path.open("rb") as handle:
        value = RestrictedFmaUnpickler(handle).load()
    if not isinstance(value, dict):
        raise TypeError("not_found.pickle is not a dictionary")
    expected = {"tracks", "albums", "artists", "audio", "clips"}
    if set(value) != expected:
        raise ValueError(f"Unexpected not_found keys: {sorted(value)}")
    return value


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def classify_license(title: Any, url: Any) -> str:
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
    restricted_markers = (
        "fma_license",
        "music-sharing",
        "sound_recording_common_law",
        "orphan_work",
    )
    if any(marker in url_text for marker in restricted_markers):
        return "EXCLUDE_RESTRICTED"
    # Includes Public Domain Mark/legacy Public Domain and non-CC licenses.
    # This is a conservative project allowlist decision, not a legal judgment.
    return "EXCLUDE_CUSTOM"


def minimum_subset_membership(value: str, scope: str) -> bool:
    if scope == "small":
        return value == "small"
    if scope == "medium_bundle":
        return value in {"small", "medium"}
    if scope == "all_curated":
        return True
    raise ValueError(scope)


def group_leakage(frame: pd.DataFrame, group_column: str) -> dict[str, int]:
    usable = frame[[group_column, "split"]].dropna()
    per_group = usable.groupby(group_column, dropna=False)["split"].nunique()
    crossing = set(per_group[per_group > 1].index)
    split_sets = {
        split: set(usable.loc[usable["split"] == split, group_column])
        for split in ("training", "validation", "test")
    }
    return {
        "unique_groups": int(usable[group_column].nunique()),
        "missing_group_rows": int(frame[group_column].isna().sum()),
        "cross_split_groups": len(crossing),
        "cross_split_affected_tracks": int(usable[group_column].isin(crossing).sum()),
        "training_validation_overlap": len(split_sets["training"] & split_sets["validation"]),
        "training_test_overlap": len(split_sets["training"] & split_sets["test"]),
        "validation_test_overlap": len(split_sets["validation"] & split_sets["test"]),
        "all_three_overlap": len(
            split_sets["training"] & split_sets["validation"] & split_sets["test"]
        ),
    }


def missingness_rows(name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        label = "|".join(map(str, column)) if isinstance(column, tuple) else str(column)
        series = frame[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "table": name,
                "column": label,
                "rows": len(frame),
                "missing": missing,
                "missing_fraction": f"{missing / len(frame):.9f}" if len(frame) else "",
                "unique_non_null": int(series.nunique(dropna=True)),
                "dtype": str(series.dtype),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    extracted = args.extracted.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    archive_before = archive.stat()

    archive_sha256 = hash_file(archive, "sha256")
    zip_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            zip_rows.append(
                {
                    "member": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "safe_path": safe_member(info.filename),
                    "compression_method": info.compress_type,
                }
            )
        corrupt_member = zf.testzip()

    expected_sha1 = parse_checksum_file(extracted / "checksums")
    extracted_rows: list[dict[str, Any]] = []
    checksum_mismatches: list[str] = []
    zip_by_name = {PurePosixPath(row["member"]).name: row for row in zip_rows}
    for path in sorted(p for p in extracted.iterdir() if p.is_file()):
        expected = expected_sha1.get(path.name, "")
        actual = hash_file(path, "sha1") if expected else "NOT_LISTED"
        checksum_status = "NOT_LISTED"
        if expected:
            checksum_status = "OK" if actual == expected else "MISMATCH"
            if checksum_status != "OK":
                checksum_mismatches.append(path.name)
        zip_row = zip_by_name.get(path.name)
        extracted_rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha1_expected": expected,
                "sha1_actual": actual,
                "checksum_status": checksum_status,
                "zip_size_match": bool(zip_row and zip_row["bytes"] == path.stat().st_size),
            }
        )

    tracks = pd.read_csv(
        extracted / "tracks.csv",
        header=[0, 1],
        index_col=0,
        low_memory=False,
    )
    tracks.index = tracks.index.astype(int)
    raw = pd.read_csv(extracted / "raw_tracks.csv", low_memory=False)
    raw["track_id"] = raw["track_id"].astype(int)
    raw_indexed = raw.set_index("track_id", drop=False)

    curated = pd.DataFrame(index=tracks.index)
    curated.index.name = "track_id"
    curated["subset"] = tracks[("set", "subset")].astype("string")
    curated["split"] = tracks[("set", "split")].astype("string")
    curated["artist_id"] = tracks[("artist", "id")]
    curated["album_id"] = tracks[("album", "id")]
    curated["artist_name"] = tracks[("artist", "name")]
    curated["album_title"] = tracks[("album", "title")]
    curated["track_title"] = tracks[("track", "title")]
    curated = curated.join(
        raw_indexed[
            [
                "license_title",
                "license_url",
                "track_file",
                "track_url",
            ]
        ],
        how="left",
    )
    curated["license_category"] = [
        classify_license(title, url)
        for title, url in zip(curated["license_title"], curated["license_url"])
    ]
    curated["allowed"] = curated["license_category"].isin(ALLOW_CATEGORIES)
    curated["small_member"] = curated["subset"].eq("small")
    curated["medium_member"] = curated["subset"].isin(["small", "medium"])

    allowlist = curated[curated["allowed"]].reset_index()
    allowlist_rows = []
    for row in allowlist.itertuples(index=False):
        allowlist_rows.append(
            {
                "track_id": row.track_id,
                "subset_label": row.subset,
                "small_member": bool(row.small_member),
                "medium_member": bool(row.medium_member),
                "split": row.split,
                "artist_id": "" if pd.isna(row.artist_id) else int(row.artist_id),
                "album_id": "" if pd.isna(row.album_id) else int(row.album_id),
                "artist_name": clean(row.artist_name),
                "album_title": clean(row.album_title),
                "track_title": clean(row.track_title),
                "license_category": row.license_category,
                "license_title": clean(row.license_title),
                "license_url": clean(row.license_url),
                "track_file": clean(row.track_file),
            }
        )
    write_csv(
        output / "fma-license-allowlist.csv",
        allowlist_rows,
        [
            "track_id",
            "subset_label",
            "small_member",
            "medium_member",
            "split",
            "artist_id",
            "album_id",
            "artist_name",
            "album_title",
            "track_title",
            "license_category",
            "license_title",
            "license_url",
            "track_file",
        ],
    )

    license_summary_rows: list[dict[str, Any]] = []
    scope_masks = {
        "all_curated": pd.Series(True, index=curated.index),
        "small": curated["subset"].eq("small"),
        "medium_bundle": curated["subset"].isin(["small", "medium"]),
    }
    for scope, mask in scope_masks.items():
        scoped = curated.loc[mask]
        by_category = scoped.groupby("license_category", dropna=False).size()
        for category, count in by_category.items():
            license_summary_rows.append(
                {
                    "record_type": "category_summary",
                    "scope": scope,
                    "license_category": category,
                    "license_title": "",
                    "license_url": "",
                    "track_count": int(count),
                    "allowed": str(category).startswith("ALLOW_"),
                }
            )
        detail = (
            scoped.groupby(
                ["license_category", "license_title", "license_url"],
                dropna=False,
            )
            .size()
            .reset_index(name="track_count")
        )
        for row in detail.itertuples(index=False):
            license_summary_rows.append(
                {
                    "record_type": "license_detail",
                    "scope": scope,
                    "license_category": row.license_category,
                    "license_title": clean(row.license_title),
                    "license_url": clean(row.license_url),
                    "track_count": int(row.track_count),
                    "allowed": str(row.license_category).startswith("ALLOW_"),
                }
            )
    write_csv(
        output / "fma-license-summary.csv",
        license_summary_rows,
        [
            "record_type",
            "scope",
            "license_category",
            "license_title",
            "license_url",
            "track_count",
            "allowed",
        ],
    )

    missingness = missingness_rows("tracks.csv", tracks) + missingness_rows(
        "raw_tracks.csv", raw
    )
    write_csv(
        output / "fma-metadata-missingness.csv",
        missingness,
        [
            "table",
            "column",
            "rows",
            "missing",
            "missing_fraction",
            "unique_non_null",
            "dtype",
        ],
    )

    not_found = load_not_found(extracted / "not_found.pickle")
    raw_ids = set(map(int, raw["track_id"]))
    curated_ids = set(map(int, tracks.index))
    missing_track_ids = set(map(int, not_found["tracks"]))
    missing_audio_ids = set(map(int, not_found["audio"]))
    missing_clip_ids = set(map(int, not_found["clips"]))

    subset_split = (
        curated.groupby(["subset", "split"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    subset_labels = curated["subset"].value_counts(dropna=False).to_dict()
    split_counts = curated["split"].value_counts(dropna=False).to_dict()

    scope_summary: dict[str, Any] = {}
    for scope, mask in scope_masks.items():
        scoped = curated.loc[mask]
        category_counts = scoped["license_category"].value_counts().to_dict()
        scope_summary[scope] = {
            "tracks": len(scoped),
            "split_counts": {
                str(key): int(value)
                for key, value in scoped["split"].value_counts().items()
            },
            "allowed_tracks": int(scoped["allowed"].sum()),
            "excluded_tracks": int((~scoped["allowed"]).sum()),
            "license_category_counts": {
                str(key): int(value) for key, value in category_counts.items()
            },
            "artist_split_leakage": group_leakage(scoped, "artist_id"),
            "album_split_leakage": group_leakage(scoped, "album_id"),
        }

    file_counts = raw["track_file"].dropna().value_counts()
    curated_file_counts = curated["track_file"].dropna().value_counts()
    small_file_counts = curated.loc[curated["small_member"], "track_file"].dropna().value_counts()

    archive_after = archive.stat()
    summary = {
        "DATA_READINESS": "BLOCKED",
        "audit_time_utc": started,
        "network_requests_made": False,
        "archive": {
            "path": str(archive),
            "bytes": archive_after.st_size,
            "sha256": archive_sha256,
            "member_count": len(zip_rows),
            "corrupt_member": corrupt_member,
            "unsafe_path_count": sum(not row["safe_path"] for row in zip_rows),
            "modified_during_audit": (
                archive_before.st_size != archive_after.st_size
                or archive_before.st_mtime_ns != archive_after.st_mtime_ns
            ),
        },
        "extracted_integrity": {
            "checksum_entries": len(expected_sha1),
            "checksum_mismatches": checksum_mismatches,
            "zip_size_mismatch_count": sum(
                not row["zip_size_match"] for row in extracted_rows
            ),
            "files": extracted_rows,
        },
        "tracks": {
            "raw_tracks": len(raw),
            "curated_tracks": len(tracks),
            "raw_not_curated": len(raw_ids - curated_ids),
            "curated_not_raw": len(curated_ids - raw_ids),
            "duplicate_raw_track_ids": int(raw["track_id"].duplicated().sum()),
            "duplicate_curated_track_ids": int(tracks.index.duplicated().sum()),
            "subset_label_counts": {
                str(key): int(value) for key, value in subset_labels.items()
            },
            "split_counts": {
                str(key): int(value) for key, value in split_counts.items()
            },
            "subset_split_counts": {
                str(subset): {
                    str(split): int(value)
                    for split, value in row.items()
                }
                for subset, row in subset_split.to_dict(orient="index").items()
            },
        },
        "not_found": {
            "tracks": len(not_found["tracks"]),
            "albums": len(not_found["albums"]),
            "artists": len(not_found["artists"]),
            "audio": len(not_found["audio"]),
            "clips": len(not_found["clips"]),
            "track_ids_unique": len(missing_track_ids),
            "track_ids_overlap_raw": len(missing_track_ids & raw_ids),
            "raw_plus_not_found_track_ids_contiguous_0_to_max": (
                min(raw_ids | missing_track_ids) == 0
                and len(raw_ids | missing_track_ids)
                == max(raw_ids | missing_track_ids) + 1
            ),
            "audio_ids_overlap_raw": len(missing_audio_ids & raw_ids),
            "audio_ids_overlap_curated": len(missing_audio_ids & curated_ids),
            "clip_ids_overlap_raw": len(missing_clip_ids & raw_ids),
            "clip_ids_overlap_curated": len(missing_clip_ids & curated_ids),
        },
        "path_duplicates": {
            "raw_missing_track_file": int(raw["track_file"].isna().sum()),
            "raw_duplicate_path_groups": int((file_counts > 1).sum()),
            "raw_duplicate_path_affected_tracks": int(file_counts[file_counts > 1].sum()),
            "raw_duplicate_path_extra_rows": int(
                (file_counts - 1).clip(lower=0).sum()
            ),
            "raw_max_path_group": int(file_counts.max()),
            "curated_missing_track_file": int(curated["track_file"].isna().sum()),
            "curated_duplicate_path_groups": int((curated_file_counts > 1).sum()),
            "curated_duplicate_path_affected_tracks": int(
                curated_file_counts[curated_file_counts > 1].sum()
            ),
            "curated_duplicate_path_extra_rows": int(
                (curated_file_counts - 1).clip(lower=0).sum()
            ),
            "small_duplicate_path_groups": int((small_file_counts > 1).sum()),
            "small_duplicate_path_affected_tracks": int(
                small_file_counts[small_file_counts > 1].sum()
            ),
        },
        "scopes": scope_summary,
        "source_currency": {
            "metadata_readme_code_url": "https://github.com/mdeff/fma",
            "metadata_readme_paper_url": "https://arxiv.org/abs/1612.01840",
            "network_verified_on_audit_date": False,
            "limitation": "URLs are provenance fields recorded in the local metadata. Current reachability, redirects, license-page changes, and download availability were not checked because this audit made no network requests.",
        },
        "blocking_reasons": [
            "This is a metadata-only audit; fma_small.zip is zero bytes and no FMA audio was audited.",
            "License categories are a conservative project allowlist based on historical metadata URLs, not a current legal or source-page verification.",
            "FMA is real music only and supplies no DeepVoice competition REAL/FAKE component labels or AI-generated music counterpart.",
            "Album IDs cross official train/validation/test splits even though artist IDs do not; lineage-safe regrouping is required before use.",
        ],
    }
    with (output / "fma-audit-run.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    write_csv(
        output / "fma-integrity-inventory.csv",
        extracted_rows,
        [
            "file",
            "bytes",
            "sha1_expected",
            "sha1_actual",
            "checksum_status",
            "zip_size_match",
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
