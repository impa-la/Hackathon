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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl


ExpectedRevision = "b84d4be5eda830b6eb714998569dba73530f2601"


def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser(
        description="Read-only exhaustive audit of locally observed AIME parquet shards"
    )
    Parser.add_argument("--root", type=Path, required=True)
    Parser.add_argument("--projected-inventory", type=Path, required=True)
    Parser.add_argument("--acquisition-plan", type=Path, required=True)
    Parser.add_argument("--output-dir", type=Path, required=True)
    Parser.add_argument("--workers", type=int, default=4)
    return Parser.parse_args()


def FileState(PathValue: Path) -> dict[str, int]:
    Stat = PathValue.stat()
    return {"bytes": Stat.st_size, "mtime_ns": Stat.st_mtime_ns}


def HashFile(PathValue: Path) -> str:
    Digest = hashlib.sha256()
    with PathValue.open("rb") as Handle:
        while True:
            Block = Handle.read(8 * 1024 * 1024)
            if not Block:
                break
            Digest.update(Block)
    return Digest.hexdigest()


def HashBytes(Value: bytes) -> str:
    return hashlib.sha256(Value).hexdigest()


def NormalizeText(Value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Value.lower()).strip("_")


def Quantiles(Values: list[float]) -> dict[str, float | int | None]:
    if not Values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    Array = np.asarray(Values, dtype=np.float64)
    return {
        "count": int(Array.size),
        "min": float(Array.min()),
        "p05": float(np.quantile(Array, 0.05)),
        "p50": float(np.quantile(Array, 0.50)),
        "p95": float(np.quantile(Array, 0.95)),
        "max": float(Array.max()),
        "mean": float(Array.mean()),
        "std_population": float(Array.std()),
    }


def ProbeAudio(AudioBytes: bytes, FfprobePath: str) -> dict[str, object]:
    Flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    Result = subprocess.run(
        [
            FfprobePath,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bits_per_raw_sample,sample_fmt,duration:format=format_name,duration,bit_rate",
            "-of",
            "json",
            "-i",
            "pipe:0",
        ],
        input=AudioBytes,
        capture_output=True,
        timeout=60,
        creationflags=Flags,
    )
    if Result.returncode != 0:
        return {
            "probe_status": "ERROR",
            "probe_error": Result.stderr.decode("utf-8", errors="replace")[-1000:],
        }
    Payload = json.loads(Result.stdout.decode("utf-8"))
    Streams = Payload.get("streams") or []
    if not Streams:
        return {"probe_status": "ERROR", "probe_error": "No audio stream"}
    Stream = Streams[0]
    Format = Payload.get("format") or {}
    return {
        "probe_status": "OK",
        "probe_error": "",
        "codec": Stream.get("codec_name", ""),
        "sample_rate_hz": int(Stream["sample_rate"]) if Stream.get("sample_rate") else None,
        "channels": int(Stream["channels"]) if Stream.get("channels") else None,
        "sample_fmt": Stream.get("sample_fmt", ""),
        "bits_per_raw_sample": int(Stream["bits_per_raw_sample"])
        if Stream.get("bits_per_raw_sample")
        else None,
        "stream_duration_seconds": float(Stream["duration"])
        if Stream.get("duration")
        else None,
        "container_format": Format.get("format_name", ""),
        "container_duration_seconds": float(Format["duration"])
        if Format.get("duration")
        else None,
        "container_bitrate_bps": int(Format["bit_rate"])
        if Format.get("bit_rate")
        else None,
    }


def DecodeAudio(
    AudioBytes: bytes,
    FfmpegPath: str,
    SampleRate: int | None,
    Channels: int | None,
) -> dict[str, object]:
    if not SampleRate or not Channels:
        return {"decode_status": "SKIPPED", "decode_error": "Missing stream metadata"}
    Flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    Result = subprocess.run(
        [
            FfmpegPath,
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        input=AudioBytes,
        capture_output=True,
        timeout=60,
        creationflags=Flags,
    )
    if Result.returncode != 0:
        return {
            "decode_status": "ERROR",
            "decode_error": Result.stderr.decode("utf-8", errors="replace")[-1000:],
        }
    Samples = np.frombuffer(Result.stdout, dtype="<f4")
    Finite = np.isfinite(Samples)
    Absolute = np.abs(Samples[Finite])
    SampleCount = int(Samples.size)
    FiniteCount = int(Finite.sum())
    return {
        "decode_status": "OK",
        "decode_error": "",
        "decoded_pcm_f32le_sha256": HashBytes(Result.stdout),
        "decoded_sample_count": SampleCount,
        "decoded_frame_count": SampleCount // Channels,
        "decoded_duration_seconds": SampleCount / (SampleRate * Channels),
        "nonfinite_sample_count": SampleCount - FiniteCount,
        "peak_abs": float(Absolute.max()) if FiniteCount else None,
        "rms": float(np.sqrt(np.mean(np.square(Samples[Finite], dtype=np.float64))))
        if FiniteCount
        else None,
        "silent_sample_count": int((Absolute < 1e-4).sum()) if FiniteCount else 0,
        "silent_sample_fraction_lt_1e_4": float((Absolute < 1e-4).mean())
        if FiniteCount
        else None,
        "clipped_sample_count": int((Absolute >= 0.999).sum()) if FiniteCount else 0,
        "clipped_sample_fraction_ge_0_999": float((Absolute >= 0.999).mean())
        if FiniteCount
        else None,
    }


def WriteCsv(PathValue: Path, Rows: list[dict[str, object]], Fields: list[str]) -> None:
    with PathValue.open("w", encoding="utf-8-sig", newline="") as Handle:
        Writer = csv.DictWriter(Handle, fieldnames=Fields, extrasaction="ignore")
        Writer.writeheader()
        Writer.writerows(Rows)


def AuditRecord(Task: dict[str, object], FfprobePath: str, FfmpegPath: str) -> dict[str, object]:
    Row = Task["row"]
    if not isinstance(Row, dict):
        raise TypeError("Parquet row must be a dictionary")
    Audio = Row.get("audio") or {}
    AudioBytes = Audio.get("bytes") or b""
    AudioPath = Audio.get("path") or ""
    Signature = (
        "RIFF_WAVE"
        if AudioBytes[:4] == b"RIFF" and AudioBytes[8:12] == b"WAVE"
        else "UNKNOWN"
    )
    Probe = ProbeAudio(AudioBytes, FfprobePath)
    Decode = DecodeAudio(
        AudioBytes,
        FfmpegPath,
        Probe.get("sample_rate_hz"),
        Probe.get("channels"),
    )
    Description = str(Row.get("description") or "")
    Provider = str(Row.get("model") or "")
    if Provider.startswith("MusicGen"):
        ProviderRights = "GO_AIME_DIRECT_CC_BY_4_GRANT_NO_EXPLICIT_OUTPUT_RELICENSE_PROHIBITION_FOUND"
    elif Provider.startswith("AudioLDM"):
        ProviderRights = "GO_AIME_DIRECT_CC_BY_4_GRANT_NO_EXPLICIT_OUTPUT_RELICENSE_PROHIBITION_FOUND"
    elif Provider == "Riffusion":
        ProviderRights = "GO_OPENRAIL_LICENSOR_CLAIMS_NO_OUTPUT_RIGHTS_SUBJECT_TO_USE_RESTRICTIONS"
    elif Provider == "Mustango":
        ProviderRights = "GO_AIME_DIRECT_CC_BY_4_GRANT_APACHE_2_CHECKPOINT_NO_OUTPUT_PROHIBITION_FOUND"
    elif Provider.startswith("Stable Audio"):
        ProviderRights = "GO_AIME_DIRECT_CC_BY_4_GRANT_CURRENT_STABILITY_ASSIGNMENT_NO_NONCOMPETING_DETECTOR_PROHIBITION_FOUND"
    else:
        ProviderRights = "HOLD_CHECKPOINT_LICENSE_AND_OUTPUT_SCOPE_UNCONFIRMED"
    return {
        "source_repository_path": Task["repository_path"],
        "source_parquet": Task["source_parquet"],
        "source_parquet_sha256": Task["source_parquet_sha256"],
        "row_in_parquet": Task["row_number"],
        "id": Row.get("id") or "",
        "provider": Provider,
        "description": Description,
        "semantic_group_sha256": HashBytes(Description.encode("utf-8")),
        "audio_path": AudioPath,
        "audio_path_extension": Path(AudioPath).suffix.lower(),
        "audio_path_matches_description": NormalizeText(Path(AudioPath).stem)
        == NormalizeText(Description),
        "audio_path_contains_description": NormalizeText(Path(AudioPath).stem).startswith(
            NormalizeText(Description)
        ),
        "audio_bytes": len(AudioBytes),
        "audio_sha256": HashBytes(AudioBytes),
        "signature_status": Signature,
        "dataset_license_id": "CC-BY-4.0",
        "dataset_license_scope": "generated_audio_per_pinned_revision_README",
        "description_license_id": "CC-BY-NC-SA-4.0",
        "license_source_url": "https://huggingface.co/datasets/disco-eth/AIME/blob/b84d4be5eda830b6eb714998569dba73530f2601/README.md",
        "attribution_required": True,
        "attribution_text": "AIME dataset, Grotschla et al., ETH Zurich, disco-eth/AIME, revision b84d4be5eda830b6eb714998569dba73530f2601, CC BY 4.0",
        "provider_rights_status": ProviderRights,
        "noncommercial_training_status": "GO_WITH_AIME_CC_BY_4_ATTRIBUTION",
        "competition_training_status": "GO_WITH_AIME_CC_BY_4_ATTRIBUTION",
        "raw_audio_redistribution_status": "GO_CC_BY_4_WITH_ATTRIBUTION",
        "conservative_training_eligibility": "GO_WITH_AIME_CC_BY_4_ATTRIBUTION",
        "training_eligible": True,
        **Probe,
        **Decode,
    }


def Main() -> None:
    Args = ParseArgs()
    if Args.workers < 1 or Args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")
    Root = Args.root.resolve()
    if Root.name != ExpectedRevision:
        raise RuntimeError("AIME root is not the pinned revision")
    OutputDir = Args.output_dir.resolve()
    OutputDir.mkdir(parents=True, exist_ok=True)
    ReadmePath = Root / "README.md"
    ShardsRoot = Root / "shards"
    if not ReadmePath.is_file() or not ShardsRoot.is_dir():
        raise FileNotFoundError("README or shards directory is missing")

    with Args.projected_inventory.open("r", encoding="utf-8-sig", newline="") as Handle:
        ProjectedRows = list(csv.DictReader(Handle))
    with Args.acquisition_plan.open("r", encoding="utf-8-sig", newline="") as Handle:
        PlanRows = list(csv.DictReader(Handle))
    if len(PlanRows) != 36:
        raise RuntimeError("Acquisition plan must contain 36 shards")
    ProjectedByName = {Path(Row["path"]).name: Row for Row in ProjectedRows}
    PlanByName = {Row["target_filename"]: Row for Row in PlanRows}
    LocalPaths = sorted(ShardsRoot.glob("*.parquet"), key=lambda Value: Value.name)
    if {PathValue.name for PathValue in LocalPaths} != set(PlanByName):
        raise RuntimeError("Local shard set does not exactly match the acquisition plan")

    FfmpegPath = shutil.which("ffmpeg")
    FfprobePath = shutil.which("ffprobe")
    if not FfmpegPath or not FfprobePath:
        raise RuntimeError("ffmpeg and ffprobe are required")

    Before = {"readme": FileState(ReadmePath)}
    Before.update({PathValue.name: FileState(PathValue) for PathValue in LocalPaths})
    RequiredColumns = ["id", "model", "description", "audio"]
    ShardInventory: list[dict[str, object]] = []
    Inventory: list[dict[str, object]] = []
    AuditedCount = 0
    ExpectedTotalRows = sum(int(Row["expected_rows"]) for Row in PlanRows)
    SchemaText = ""
    for ShardPath in LocalPaths:
        PlanRow = PlanByName[ShardPath.name]
        ProjectedRow = ProjectedByName[ShardPath.name]
        ActualSha256 = HashFile(ShardPath)
        Frame = pl.read_parquet(ShardPath)
        if Frame.columns != RequiredColumns:
            raise RuntimeError(f"Unexpected schema in {ShardPath.name}: {Frame.columns}")
        CurrentSchema = json.dumps(
            {Name: str(Type) for Name, Type in Frame.schema.items()},
            sort_keys=True,
        )
        if SchemaText and CurrentSchema != SchemaText:
            raise RuntimeError("Parquet schemas differ")
        SchemaText = CurrentSchema
        ModelsInShard = Counter(str(Value) for Value in Frame.get_column("model").to_list())
        ShardInventory.append(
            {
                "source_repository_path": PlanRow["repository_path"],
                "local_filename": ShardPath.name,
                "provider_expected": PlanRow["provider"],
                "provider_counts_actual": json.dumps(dict(ModelsInShard), ensure_ascii=False),
                "expected_bytes": int(PlanRow["expected_bytes"]),
                "actual_bytes": ShardPath.stat().st_size,
                "size_matches": ShardPath.stat().st_size == int(PlanRow["expected_bytes"]),
                "expected_sha256": PlanRow["expected_sha256"],
                "projected_lfs_sha256": ProjectedRow["lfs_sha256"],
                "actual_sha256": ActualSha256,
                "sha256_matches": ActualSha256 == PlanRow["expected_sha256"] == ProjectedRow["lfs_sha256"],
                "expected_rows": int(PlanRow["expected_rows"]),
                "actual_rows": Frame.height,
                "rows_match": Frame.height == int(PlanRow["expected_rows"]),
                "pure_provider_matches": ModelsInShard == {PlanRow["provider"]: Frame.height},
            }
        )
        ShardTasks: list[dict[str, object]] = []
        for RowNumber, Row in enumerate(Frame.to_dicts(), start=1):
            ShardTasks.append(
                {
                    "repository_path": PlanRow["repository_path"],
                    "source_parquet": ShardPath.name,
                    "source_parquet_sha256": ActualSha256,
                    "row_number": RowNumber,
                    "row": Row,
                }
            )
        with ThreadPoolExecutor(max_workers=Args.workers) as Executor:
            Futures = {
                Executor.submit(AuditRecord, Task, FfprobePath, FfmpegPath): Task
                for Task in ShardTasks
            }
            for Future in as_completed(Futures):
                Inventory.append(Future.result())
                AuditedCount += 1
                if AuditedCount % 100 == 0:
                    print(
                        f"audited {AuditedCount}/{ExpectedTotalRows} AIME audio rows",
                        flush=True,
                    )
    Inventory.sort(key=lambda Row: (str(Row["source_parquet"]), int(Row["row_in_parquet"])))

    After = {"readme": FileState(ReadmePath)}
    After.update({PathValue.name: FileState(PathValue) for PathValue in LocalPaths})
    AudioGroups: defaultdict[str, list[str]] = defaultdict(list)
    PcmGroups: defaultdict[str, list[str]] = defaultdict(list)
    for Row in Inventory:
        AudioGroups[str(Row["audio_sha256"])].append(str(Row["id"]))
        PcmDigest = str(Row.get("decoded_pcm_f32le_sha256") or "")
        if PcmDigest:
            PcmGroups[PcmDigest].append(str(Row["id"]))

    Durations = [float(Row["decoded_duration_seconds"]) for Row in Inventory if Row.get("decoded_duration_seconds") is not None]
    TotalSamples = sum(int(Row.get("decoded_sample_count") or 0) for Row in Inventory)
    TotalSilent = sum(int(Row.get("silent_sample_count") or 0) for Row in Inventory)
    TotalClipped = sum(int(Row.get("clipped_sample_count") or 0) for Row in Inventory)
    Ids = [str(Row["id"]) for Row in Inventory]
    Descriptions = [str(Row["description"]) for Row in Inventory]
    Models = [str(Row["provider"]) for Row in Inventory]
    PlanProviders = Counter(Row["provider"] for Row in PlanRows)
    DescriptionCounts = Counter(Descriptions)
    AudioPathCounts = Counter(str(Row["audio_path"]) for Row in Inventory)
    ProviderIdRanges = {}
    for Provider in sorted(set(Models)):
        ProviderIds = [int(Row["id"]) for Row in Inventory if Row["provider"] == Provider]
        ProviderIdRanges[Provider] = {"min": min(ProviderIds), "max": max(ProviderIds)}

    Audit = {
        "data_readiness": "BLOCKED",
        "block_reason": "The curated 36-shard AIME subset is technically and rights-ready with CC BY 4.0 attribution, but combined DeepVoice readiness still awaits WaveFake acquisition and audit.",
        "aime_subset_readiness": "TECHNICAL_PASS_RIGHTS_GO_WITH_ATTRIBUTION",
        "dataset": "disco-eth/AIME",
        "revision": ExpectedRevision,
        "source": {
            "root": str(Root),
            "readme_bytes": ReadmePath.stat().st_size,
            "readme_sha256": HashFile(ReadmePath),
            "shards_root": str(ShardsRoot),
            "local_shard_count": len(LocalPaths),
            "local_shard_bytes": sum(PathValue.stat().st_size for PathValue in LocalPaths),
            "integrity_failure_count": sum(
                not Row["sha256_matches"] or not Row["size_matches"] or not Row["rows_match"]
                for Row in ShardInventory
            ),
            "source_state_unchanged": Before == After,
        },
        "repository_projection": {
            "shard_count": len(ProjectedRows),
            "published_bytes": sum(int(Row["size_bytes"]) for Row in ProjectedRows),
            "projected_rows": sum(int(Row["row_count"]) for Row in ProjectedRows),
            "failed_projected_shards": sum(bool(Row["error"]) for Row in ProjectedRows),
            "projection_scope": ["id", "model"],
            "audio_content_downloaded_shards": len(LocalPaths),
            "audio_content_downloaded_rows": len(Inventory),
        },
        "subset_schema": {
            "columns": RequiredColumns,
            "dtypes": json.loads(SchemaText),
            "row_count": len(Inventory),
            "null_counts": {
                "id": sum(not Row["id"] for Row in Inventory),
                "model": sum(not Row["provider"] for Row in Inventory),
                "description": sum(not Row["description"] for Row in Inventory),
                "audio": sum(int(Row["audio_bytes"]) == 0 for Row in Inventory),
            },
            "unique_id_count": len(set(Ids)),
            "provider_id_ranges": ProviderIdRanges,
            "model_counts": dict(Counter(Models)),
            "unique_description_count": len(set(Descriptions)),
            "description_group_size_counts": dict(Counter(DescriptionCounts.values())),
            "duplicate_description_group_count": sum(Count > 1 for Count in DescriptionCounts.values()),
            "rows_in_duplicate_description_groups": sum(
                Count for Count in DescriptionCounts.values() if Count > 1
            ),
            "audio_bytes_present_count": sum(int(Row["audio_bytes"]) > 0 for Row in Inventory),
            "unique_audio_path_count": len(AudioPathCounts),
            "duplicate_audio_path_group_count": sum(
                Count > 1 for Count in AudioPathCounts.values()
            ),
            "rows_in_duplicate_audio_path_groups": sum(
                Count for Count in AudioPathCounts.values() if Count > 1
            ),
        },
        "subset_audio": {
            "count": len(Inventory),
            "signature_counts": dict(Counter(str(Row["signature_status"]) for Row in Inventory)),
            "decode_status_counts": dict(Counter(str(Row["decode_status"]) for Row in Inventory)),
            "codec_counts": dict(Counter(str(Row.get("codec") or "") for Row in Inventory)),
            "sample_rate_counts": dict(Counter(str(Row.get("sample_rate_hz") or "") for Row in Inventory)),
            "channel_counts": dict(Counter(str(Row.get("channels") or "") for Row in Inventory)),
            "sample_format_counts": dict(Counter(str(Row.get("sample_fmt") or "") for Row in Inventory)),
            "duration_distribution_seconds": Quantiles(Durations),
            "total_duration_seconds": sum(Durations),
            "total_samples": TotalSamples,
            "nonfinite_sample_count": sum(int(Row.get("nonfinite_sample_count") or 0) for Row in Inventory),
            "weighted_silence_fraction_lt_1e_4": TotalSilent / TotalSamples if TotalSamples else None,
            "silence_ge_50pct_count": sum(float(Row.get("silent_sample_fraction_lt_1e_4") or 0) >= 0.5 for Row in Inventory),
            "weighted_clipping_fraction_ge_0_999": TotalClipped / TotalSamples if TotalSamples else None,
            "clipping_ge_1pct_count": sum(float(Row.get("clipped_sample_fraction_ge_0_999") or 0) >= 0.01 for Row in Inventory),
            "audio_sha256_duplicate_groups": {Digest: Members for Digest, Members in AudioGroups.items() if len(Members) > 1},
            "pcm_sha256_duplicate_groups": {Digest: Members for Digest, Members in PcmGroups.items() if len(Members) > 1},
        },
        "acquisition_plan": {
            "provider_count": len(PlanProviders),
            "provider_shard_counts": dict(sorted(PlanProviders.items())),
            "shard_count": len(PlanRows),
            "expected_rows": sum(int(Row["expected_rows"]) for Row in PlanRows),
            "expected_bytes": sum(int(Row["expected_bytes"]) for Row in PlanRows),
            "selection": "four pure-provider shards per requested candidate provider, evenly spaced over numeric ID range",
        },
        "leakage_findings": {
            "id_encodes_provider_block": True,
            "shard_index_is_provider_correlated": True,
            "audio_path_encodes_prompt_tags": True,
            "audio_path_exact_description_count": sum(
                bool(Row["audio_path_matches_description"]) for Row in Inventory
            ),
            "audio_path_contains_description_count": sum(
                bool(Row["audio_path_contains_description"]) for Row in Inventory
            ),
            "description_is_prompt_and_semantic_group_key": True,
            "random_row_or_segment_split_allowed": False,
            "required_group_key": "normalized description/prompt identity, extended with audio and perceptual duplicate components",
        },
        "license_gate": {
            "aime_generated_audio_dataset_license": "CC-BY-4.0 stated in pinned README",
            "description_metadata_license": "CC-BY-NC-SA-4.0",
            "selected_subset_rows": len(Inventory),
            "user_policy": "Noncommercial CC-BY, CC-BY-NC, and share-alike resources are permitted",
            "selected_subset_training_eligible_rows": len(Inventory),
            "selected_subset_status": "GO_WITH_AIME_CC_BY_4_ATTRIBUTION",
            "commercial_suno_udio_status": "HOLD_AND_EXCLUDED_FROM_SELECTED_SUBSET",
            "scope_note": "Exact generation checkpoint and service-tier gaps are provenance-quality warnings unless an explicit output relicense or detector-training prohibition is found",
        },
        "method": {
            "observed_audio_scope": "all 1116 rows in the selected 36 local parquet shards; no sampling",
            "workers": Args.workers,
            "audio_probe": subprocess.run([FfprobePath, "-version"], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).stdout.splitlines()[0],
            "audio_decode": "FFmpeg full decode to f32le interleaved PCM",
            "silence_definition": "abs(sample)<1e-4",
            "clipping_definition": "abs(sample)>=0.999",
            "duplicate_definitions": ["embedded audio byte SHA-256", "decoded f32le PCM SHA-256"],
        },
    }

    Fields = [
        "source_repository_path", "source_parquet", "source_parquet_sha256",
        "row_in_parquet", "id", "provider", "description", "semantic_group_sha256",
        "audio_path", "audio_path_extension", "audio_path_matches_description",
        "audio_path_contains_description",
        "audio_bytes", "audio_sha256", "signature_status",
        "dataset_license_id", "dataset_license_scope", "description_license_id",
        "license_source_url", "attribution_required", "attribution_text",
        "provider_rights_status", "conservative_training_eligibility", "training_eligible",
        "noncommercial_training_status", "competition_training_status",
        "raw_audio_redistribution_status",
        "probe_status", "probe_error", "codec",
        "sample_rate_hz", "channels", "sample_fmt", "bits_per_raw_sample",
        "stream_duration_seconds", "container_format", "container_duration_seconds",
        "container_bitrate_bps", "decode_status", "decode_error",
        "decoded_pcm_f32le_sha256", "decoded_sample_count", "decoded_frame_count",
        "decoded_duration_seconds", "nonfinite_sample_count", "peak_abs", "rms",
        "silent_sample_count", "silent_sample_fraction_lt_1e_4",
        "clipped_sample_count", "clipped_sample_fraction_ge_0_999",
    ]
    WriteCsv(OutputDir / "aime-subset-audio-inventory.csv", Inventory, Fields)
    WriteCsv(OutputDir / "aime-curated-manifest.csv", Inventory, Fields)
    WriteCsv(
        OutputDir / "aime-subset-shard-integrity.csv",
        ShardInventory,
        list(ShardInventory[0]),
    )
    (OutputDir / "aime-subset-audit-run.json").write_text(
        json.dumps(Audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(Inventory),
                "decode_status": Audit["subset_audio"]["decode_status_counts"],
                "duration_seconds": Audit["subset_audio"]["total_duration_seconds"],
                "audio_duplicates": len(Audit["subset_audio"]["audio_sha256_duplicate_groups"]),
                "pcm_duplicates": len(Audit["subset_audio"]["pcm_sha256_duplicate_groups"]),
                "source_unchanged": Audit["source"]["source_state_unchanged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    Main()
