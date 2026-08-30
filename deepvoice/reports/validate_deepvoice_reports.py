#!/usr/bin/env python3
"""Cross-check final DeepVoice reports and deterministic gzip artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def Hash(PathValue: Path) -> str:
    Digest = hashlib.sha256()
    with PathValue.open("rb") as Stream:
        while Block := Stream.read(1024 * 1024):
            Digest.update(Block)
    return Digest.hexdigest()


def GzipMtime(PathValue: Path) -> int:
    with PathValue.open("rb") as Stream:
        Header = Stream.read(10)
    if Header[:2] != b"\x1f\x8b":
        raise RuntimeError(f"not gzip: {PathValue}")
    return int.from_bytes(Header[4:8], "little")


def Main() -> None:
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--directory", type=Path, required=True)
    Args = Parser.parse_args()
    Directory = Args.directory.resolve()
    Failures: list[str] = []

    FinalRun = json.loads((Directory / "deepvoice-final-audit-run.json").read_text(encoding="utf-8"))
    WaveRun = json.loads((Directory / "wavefake-audit-run.json").read_text(encoding="utf-8"))
    ManifestPath = Directory / "deepvoice-training-manifest.csv.gz"
    InventoryPath = Directory / "wavefake-audio-inventory.csv.gz"

    DatasetCounts: Counter[str] = Counter()
    LabelCounts: Counter[str] = Counter()
    SplitCounts: Counter[str] = Counter()
    ManifestRows = 0
    with gzip.open(ManifestPath, "rt", encoding="utf-8-sig", newline="") as Stream:
        for Row in csv.DictReader(Stream):
            ManifestRows += 1
            DatasetCounts[Row["dataset"]] += 1
            LabelCounts[Row["label"]] += 1
            SplitCounts[Row["recommended_content_split"]] += 1

    RoleCounts: Counter[str] = Counter()
    EligibleCounts: Counter[str] = Counter()
    InventoryRows = 0
    with gzip.open(InventoryPath, "rt", encoding="utf-8-sig", newline="") as Stream:
        for Row in csv.DictReader(Stream):
            InventoryRows += 1
            RoleCounts[Row["role"]] += 1
            if Row["training_eligible"].casefold() == "true":
                EligibleCounts[Row["role"]] += 1

    ExpectedDataset = {
        "ljspeech-1.1": 13_100,
        "wavefake-1.2.0": 117_983,
        "fma-small": 5_129,
        "aime-open-model-subset": 1_116,
    }
    Checks = {
        "final_ready": FinalRun["data_readiness"] == "READY",
        "wavefake_ready": WaveRun["data_readiness"] == "READY",
        "manifest_rows": ManifestRows == 137_328,
        "dataset_counts": dict(DatasetCounts) == ExpectedDataset,
        "label_counts": dict(LabelCounts) == {"real": 18_229, "synthetic": 119_099},
        "split_counts": dict(SplitCounts) == {"train": 110_059, "test": 13_729, "validation": 13_540},
        "content_group_cross_split_zero": FinalRun["manifest"]["content_groups_crossing_split_count"] == 0,
        "manifest_sha256": Hash(ManifestPath) == "2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6",
        "inventory_rows": InventoryRows == 134_266,
        "inventory_roles": dict(RoleCounts) == {"generated": 117_983, "redundant_duplicate_copy": 16_283},
        "eligible_roles": dict(EligibleCounts) == {"generated": 117_983},
        "wavefake_crc_zero": WaveRun["integrity"]["crc_failure_count"] == 0,
        "wavefake_parse_zero": WaveRun["integrity"]["parse_failure_count"] == 0,
        "wavefake_duplicate_exact": WaveRun["pairing"]["common_voice_duplicate_paths_exact_file_and_pcm_match"],
        "final_report_first_line": (Directory / "deepvoice-final-data-readiness.md").read_text(encoding="utf-8").splitlines()[0] == "DATA_READINESS: READY",
        "wavefake_report_first_line": (Directory / "wavefake-audit.md").read_text(encoding="utf-8").splitlines()[0] == "DATA_READINESS: READY",
    }
    GzipFiles = sorted(Directory.glob("*.csv.gz"))
    Checks["all_gzip_mtime_zero"] = all(GzipMtime(PathValue) == 0 for PathValue in GzipFiles)
    Checks["no_file_ge_50_mib_in_publish_set"] = all(
        PathValue.stat().st_size < 50 * 1024 * 1024
        for PathValue in Directory.iterdir()
        if PathValue.is_file() and not (PathValue.suffix == ".csv" and PathValue.stat().st_size >= 50 * 1024 * 1024)
    )
    Failures.extend(Name for Name, Passed in Checks.items() if not Passed)
    Result = {
        "status": "PASS" if not Failures else "FAIL",
        "checks": Checks,
        "failures": Failures,
        "observed": {
            "manifest_rows": ManifestRows,
            "dataset_counts": dict(DatasetCounts),
            "label_counts": dict(LabelCounts),
            "split_counts": dict(SplitCounts),
            "inventory_rows": InventoryRows,
            "role_counts": dict(RoleCounts),
            "eligible_role_counts": dict(EligibleCounts),
            "gzip_files": {PathValue.name: PathValue.stat().st_size for PathValue in GzipFiles},
        },
    }
    (Directory / "deepvoice-final-qa-run.json").write_text(
        json.dumps(Result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(Result, ensure_ascii=False, indent=2))
    if Failures:
        raise SystemExit(1)


if __name__ == "__main__":
    Main()
