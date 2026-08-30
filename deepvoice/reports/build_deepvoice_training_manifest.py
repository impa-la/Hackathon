#!/usr/bin/env python3
"""Build the rights- and leakage-aware DeepVoice training manifest.

All inputs are read-only.  The output contains eligible, technically valid rows
only; exclusions remain documented in the upstream audit inventories and run
JSON files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SplitSalt = "deepvoice-final-content-group-split-v1"
Fields = [
    "dataset",
    "label",
    "sample_id",
    "source_family",
    "generator_or_provider",
    "content_group_key",
    "recommended_content_split",
    "provider_holdout_group",
    "source_locator",
    "codec",
    "sample_rate_hz",
    "channels",
    "duration_seconds",
    "file_sha256",
    "pcm_sha256",
    "license_id",
    "license_source",
    "attribution_required",
    "sharealike_required",
    "noncommercial_only",
    "changes_notice_required",
    "attribution_text",
    "technical_status",
    "training_eligible",
    "eligibility_reason",
    "leakage_notes",
]


def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--reports-dir", type=Path, required=True)
    Parser.add_argument("--wavefake-inventory", type=Path, required=True)
    Parser.add_argument("--wavefake-archive", type=Path, required=True)
    Parser.add_argument("--ljspeech-root", type=Path, required=True)
    Parser.add_argument("--fma-root", type=Path, required=True)
    Parser.add_argument("--aime-root", type=Path, required=True)
    Parser.add_argument("--output-dir", type=Path, required=True)
    return Parser.parse_args()


def LoadCsv(PathValue: Path) -> list[dict[str, str]]:
    Opener = gzip.open if PathValue.suffix.casefold() == ".gz" else Path.open
    with Opener(PathValue, "rt", encoding="utf-8-sig", newline="") as Stream:
        return list(csv.DictReader(Stream))


def WriteDeterministicGzip(Source: Path, Target: Path) -> None:
    with Source.open("rb") as Input, Target.open("wb") as RawOutput:
        with gzip.GzipFile(filename="", mode="wb", fileobj=RawOutput, compresslevel=9, mtime=0) as Output:
            while True:
                Block = Input.read(1024 * 1024)
                if not Block:
                    break
                Output.write(Block)


def StableSplit(GroupKey: str) -> str:
    Digest = hashlib.sha256(f"{SplitSalt}|{GroupKey}".encode("utf-8")).digest()
    Value = int.from_bytes(Digest[:8], "big") / 2**64
    if Value < 0.8:
        return "train"
    if Value < 0.9:
        return "validation"
    return "test"


def Boolean(Value: object) -> bool:
    return str(Value).casefold() == "true"


def AddCommon(
    Row: dict[str, object],
    Dataset: str,
    Label: str,
    SampleId: str,
    Family: str,
    Provider: str,
    GroupKey: str,
) -> dict[str, object]:
    Row.update(
        {
            "dataset": Dataset,
            "label": Label,
            "sample_id": SampleId,
            "source_family": Family,
            "generator_or_provider": Provider,
            "content_group_key": GroupKey,
            "recommended_content_split": StableSplit(GroupKey),
            "provider_holdout_group": f"{Dataset}:{Provider}",
            "technical_status": "PASS",
            "training_eligible": True,
        }
    )
    return Row


def Main() -> None:
    Args = ParseArgs()
    ReportsDir = Args.reports_dir.resolve()
    OutputDir = Args.output_dir.resolve()
    OutputDir.mkdir(parents=True, exist_ok=True)
    OutputPath = OutputDir / "deepvoice-training-manifest.csv"
    SummaryPath = OutputDir / "deepvoice-training-manifest-summary.csv"
    AuditPath = OutputDir / "deepvoice-final-audit-run.json"

    Rows: list[dict[str, object]] = []

    LjsRows = LoadCsv(ReportsDir / "ljspeech-audio-inventory.csv")
    for Source in LjsRows:
        if Source["parse_status"] != "OK" or not Boolean(Source["metadata_present"]):
            continue
        SampleId = Source["id"]
        GroupKey = f"ljspeech:{SampleId}"
        Rows.append(
            AddCommon(
                {
                    "source_locator": str((Args.ljspeech_root / Source["relative_path"]).resolve()),
                    "codec": "pcm_s16le",
                    "sample_rate_hz": Source["sample_rate_hz"],
                    "channels": Source["channels"],
                    "duration_seconds": Source["duration_seconds"],
                    "file_sha256": Source["file_sha256"],
                    "pcm_sha256": Source["pcm_sha256"],
                    "license_id": "Public-Domain",
                    "license_source": "LJSpeech-1.1/README",
                    "attribution_required": False,
                    "sharealike_required": False,
                    "noncommercial_only": False,
                    "changes_notice_required": False,
                    "attribution_text": "",
                    "eligibility_reason": "GO_LJSPEECH_PUBLIC_DOMAIN_TECHNICAL_PASS",
                    "leakage_notes": "Keep with all WaveFake LJS variants by LJSpeech ID; transcript duplicates must not cross folds.",
                },
                "ljspeech-1.1",
                "real",
                SampleId,
                "ljspeech",
                "ljspeech_real",
                GroupKey,
            )
        )

    WaveRows = LoadCsv(Args.wavefake_inventory.resolve())
    for Source in WaveRows:
        if Source["role"] != "generated" or not Boolean(Source["training_eligible"]):
            continue
        GroupKey = Source["content_group_key"]
        Rows.append(
            AddCommon(
                {
                    "source_locator": f"zip://{Args.wavefake_archive.resolve()}!/{Source['zip_member']}",
                    "codec": Source["codec"],
                    "sample_rate_hz": Source["sample_rate_hz"],
                    "channels": Source["channels"],
                    "duration_seconds": Source["duration_seconds"],
                    "file_sha256": Source["file_sha256"],
                    "pcm_sha256": Source["pcm_sha256"],
                    "license_id": "CC-BY-SA-4.0",
                    "license_source": "WaveFake-1.2.0/LICENSE",
                    "attribution_required": True,
                    "sharealike_required": True,
                    "noncommercial_only": False,
                    "changes_notice_required": True,
                    "attribution_text": "WaveFake, Joel Frank and Lea Schoenherr, CC BY-SA 4.0, DOI 10.5281/zenodo.5642694",
                    "eligibility_reason": "GO_WAVEFAKE_CC_BY_SA_4_AFTER_EXACT_DEDUPLICATION",
                    "leakage_notes": "Split by source ID; never split segments. Common Voice-prompt nested duplicate copies are excluded upstream.",
                },
                "wavefake-1.2.0",
                "synthetic",
                f"{Source['dataset_dir']}:{Source['source_id']}",
                Source["source_family"],
                Source["generator"],
                GroupKey,
            )
        )

    FmaRows = LoadCsv(ReportsDir / "fma-small-audio-inventory.csv")
    for Source in FmaRows:
        if not Boolean(Source["strict_allowlist"]) or Source["decode_status"] != "OK":
            continue
        SampleId = Source["track_id"]
        GroupKey = f"fma:{Source['component_id']}"
        Rows.append(
            AddCommon(
                {
                    "source_locator": str((Args.fma_root / Source["relative_path"]).resolve()),
                    "codec": Source["codec"],
                    "sample_rate_hz": Source["sample_rate_hz"],
                    "channels": Source["channels"],
                    "duration_seconds": Source["decoded_duration_seconds"],
                    "file_sha256": Source["file_sha256"],
                    "pcm_sha256": Source["decoded_pcm_sha256"],
                    "license_id": Source["license_title"],
                    "license_source": Source["license_url"],
                    "attribution_required": Source["license_category"] != "ALLOW_CC0",
                    "sharealike_required": "_SA" in Source["license_category"],
                    "noncommercial_only": "_NC" in Source["license_category"],
                    "changes_notice_required": Source["license_category"] != "ALLOW_CC0",
                    "attribution_text": f"FMA track {SampleId}: {Source['artist_id']} / {Source['album_id']} / {Source['track_title']}",
                    "eligibility_reason": "GO_FMA_STRICT_ALLOWLIST_AND_DECODE_PASS",
                    "leakage_notes": "Use artist-album connected component as the indivisible split group; ignore the original row-level split when it conflicts.",
                },
                "fma-small",
                "real",
                SampleId,
                "fma_music",
                "fma_real",
                GroupKey,
            )
        )

    AimeRows = LoadCsv(ReportsDir / "aime-curated-manifest.csv")
    for Source in AimeRows:
        if not Boolean(Source["training_eligible"]) or Source["decode_status"] != "OK":
            continue
        SampleId = Source["id"]
        GroupKey = f"aime-prompt:{Source['semantic_group_sha256']}"
        Locator = (
            f"parquet://{(Args.aime_root / 'shards' / Source['source_parquet']).resolve()}"
            f"#row={Source['row_in_parquet']}"
        )
        Rows.append(
            AddCommon(
                {
                    "source_locator": Locator,
                    "codec": Source["codec"],
                    "sample_rate_hz": Source["sample_rate_hz"],
                    "channels": Source["channels"],
                    "duration_seconds": Source["decoded_duration_seconds"],
                    "file_sha256": Source["audio_sha256"],
                    "pcm_sha256": Source["decoded_pcm_f32le_sha256"],
                    "license_id": "CC-BY-4.0",
                    "license_source": Source["license_source_url"],
                    "attribution_required": True,
                    "sharealike_required": False,
                    "noncommercial_only": False,
                    "changes_notice_required": True,
                    "attribution_text": Source["attribution_text"],
                    "eligibility_reason": "GO_AIME_DIRECT_CC_BY_4_ATTRIBUTION_TECHNICAL_PASS",
                    "leakage_notes": "Split by normalized prompt/description identity; provider and shard/ID are shortcut-prone and require held-out evaluation.",
                },
                "aime-open-model-subset",
                "synthetic",
                SampleId,
                "aime_music",
                Source["provider"],
                GroupKey,
            )
        )

    with OutputPath.open("w", encoding="utf-8-sig", newline="") as Stream:
        Writer = csv.DictWriter(Stream, fieldnames=Fields)
        Writer.writeheader()
        Writer.writerows(Rows)
    CompressedOutputPath = OutputPath.with_suffix(OutputPath.suffix + ".gz")
    WriteDeterministicGzip(OutputPath, CompressedOutputPath)

    DatasetCounts = Counter(Row["dataset"] for Row in Rows)
    LabelCounts = Counter(Row["label"] for Row in Rows)
    SplitCounts = Counter(Row["recommended_content_split"] for Row in Rows)
    DatasetSplitCounts = Counter(
        (Row["dataset"], Row["label"], Row["recommended_content_split"]) for Row in Rows
    )
    SummaryRows: list[dict[str, object]] = []
    for (Dataset, Label, Split), Count in sorted(DatasetSplitCounts.items()):
        SummaryRows.append(
            {
                "dataset": Dataset,
                "label": Label,
                "recommended_content_split": Split,
                "row_count": Count,
                "duration_hours": sum(
                    float(Row["duration_seconds"])
                    for Row in Rows
                    if Row["dataset"] == Dataset
                    and Row["label"] == Label
                    and Row["recommended_content_split"] == Split
                )
                / 3600,
                "unique_content_group_count": len(
                    {
                        Row["content_group_key"]
                        for Row in Rows
                        if Row["dataset"] == Dataset
                        and Row["label"] == Label
                        and Row["recommended_content_split"] == Split
                    }
                ),
            }
        )
    with SummaryPath.open("w", encoding="utf-8-sig", newline="") as Stream:
        Writer = csv.DictWriter(Stream, fieldnames=list(SummaryRows[0]))
        Writer.writeheader()
        Writer.writerows(SummaryRows)

    GroupSplits: defaultdict[str, set[str]] = defaultdict(set)
    for Row in Rows:
        GroupSplits[str(Row["content_group_key"])].add(str(Row["recommended_content_split"]))
    ManifestBytes = OutputPath.stat().st_size
    ManifestSha256 = hashlib.sha256(OutputPath.read_bytes()).hexdigest()
    CompressedManifestBytes = CompressedOutputPath.stat().st_size
    CompressedManifestSha256 = hashlib.sha256(CompressedOutputPath.read_bytes()).hexdigest()
    Audit = {
        "data_readiness": "READY",
        "block_reason": "",
        "readiness_scope": "user-approved noncommercial hackathon training and local validation; excludes raw-corpus redistribution, commercial use, and a legal conclusion on public model-weight licensing",
        "target_definition": "binary real versus synthetic provenance label at one source-audio-asset per row; any derived segment must inherit the source row split",
        "scope": "curated technically valid and rights-eligible LJSpeech, WaveFake, FMA-small, and AIME open-model subset rows",
        "manifest": {
            "path": "deepvoice/reports/deepvoice-training-manifest.csv.gz",
            "build_path": str(CompressedOutputPath),
            "compression": "deterministic gzip, mtime=0",
            "bytes": CompressedManifestBytes,
            "sha256": CompressedManifestSha256,
            "uncompressed_bytes": ManifestBytes,
            "uncompressed_sha256": ManifestSha256,
            "row_count": len(Rows),
            "dataset_counts": dict(DatasetCounts),
            "label_counts": dict(LabelCounts),
            "split_counts": dict(SplitCounts),
            "unique_content_group_count": len(GroupSplits),
            "content_groups_crossing_split_count": sum(len(Value) > 1 for Value in GroupSplits.values()),
            "total_duration_hours": sum(float(Row["duration_seconds"]) for Row in Rows) / 3600,
        },
        "required_constraints": {
            "content_group_split": "mandatory; random row or segment split is forbidden",
            "ljspeech": "real and all seven WaveFake vocoder outputs share one LJSpeech-ID group",
            "jsut": "both WaveFake generators share one JSUT basic5000-ID group",
            "common_voice_prompt": "retain one canonical generated path per ID; exclude 16283 nested exact duplicate copies",
            "fma": "artist-album bipartite connected component is the indivisible group",
            "aime": "normalized prompt/description semantic group is indivisible",
            "evaluation": "report provider-held-out and source-family-held-out results in addition to the content split",
            "primary_paired_benchmark": "LJSpeech real 13100 versus same-ID WaveFake LJS synthetic 91700; group by ID and use class-balanced sampling/metrics",
            "confounding_warning": "FMA is real-only while AIME, JSUT, and TTS are synthetic-only; treat unmatched families as stress tests, not a single fair benchmark",
            "shortcut_control": "normalize codec/rate/channels only within each fold and report unnormalized source-stratified results",
        },
        "exclusions": {
            "wavefake_redundant_duplicate_copy": 16_283,
            "fma_small_not_strict_allowlist_or_decode_failure": 2_871,
            "fma_strict_allowlist_decode_failure": 1,
            "aime_suno_udio": "not acquired and not present in curated subset",
            "echoes": "not included because provider output-rights evidence remains conditional/HOLD",
        },
    }
    AuditPath.write_text(json.dumps(Audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(Audit["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    Main()
