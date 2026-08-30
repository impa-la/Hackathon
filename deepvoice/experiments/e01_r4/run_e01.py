# /// <summary>
# E01-R4 cache preparation, strict numerical tests and benchmark-only full-run gate
# /// </summary>

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark import (
    BenchmarkCachedEndToEndPilot,
    BenchmarkGpuBatch,
    BenchmarkWindowsWorkers,
    ProjectRuntime,
)
from .cache import (
    AuditAimeLocatorResolution,
    BuildExactCache,
    EstimateCacheStorage,
    VerifyCompleteCache,
)
from .contract_adapter import HashFile
from .determinism import ConfigureParentCpuThreads
from .preflight import RunPreflight
from .records import LoadE01Records
from .run_tests import RunAllTests, RunTinySmoke
from .sampling import SummarizeSampler
from .strict_serialization import AssertFinitePayload, JsonBytes


def RunCommand(Command: list[str], WorkingDirectory: Path) -> str:
    Result = subprocess.run(
        Command,
        cwd=WorkingDirectory,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return Result.stdout.strip() or Result.stderr.strip()


def WriteBytesExclusive(OutputPath: Path, Payload: bytes) -> None:
    if OutputPath.exists():
        raise FileExistsError(f"Refusing to overwrite E01-R4 output: {OutputPath}")
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    OutputPath.write_bytes(Payload)


def BuildCodeInventory(E01Root: Path) -> list[dict[str, Any]]:
    Rows = []
    for FilePath in sorted(E01Root.glob("*")):
        if not FilePath.is_file() or FilePath.suffix == ".pyc":
            continue
        Rows.append(
            {
                "relative_path": FilePath.name,
                "bytes": FilePath.stat().st_size,
                "sha256": HashFile(FilePath),
            }
        )
    return Rows


def InventoryDigest(Rows: list[dict[str, Any]]) -> str:
    Payload = "".join(
        f"{Row['relative_path']}\0{Row['bytes']}\0{Row['sha256']}\n" for Row in Rows
    ).encode("utf-8")
    return hashlib.sha256(Payload).hexdigest()


def CsvBytes(Rows: list[dict[str, Any]]) -> bytes:
    if not Rows:
        raise ValueError("Cannot serialize an empty CSV")
    AssertFinitePayload(Rows)
    Buffer = io.StringIO(newline="")
    Writer = csv.DictWriter(Buffer, fieldnames=list(Rows[0]))
    Writer.writeheader()
    Writer.writerows(Rows)
    return Buffer.getvalue().encode("utf-8")


def BuildReport(
    Status: str,
    Preflight: dict[str, Any],
    Tests: dict[str, Any],
    StorageEstimate: dict[str, Any],
    AimeResolution: dict[str, Any],
    CacheBuild: dict[str, Any],
    CacheGate: dict[str, Any],
    WorkerBenchmark: dict[str, Any],
    GpuAutotune: dict[str, Any],
    Pilot: dict[str, Any],
    Projection: dict[str, Any],
    RunManifest: dict[str, Any],
) -> str:
    Lines = [
        f"EXPERIMENT_BATCH: {Status}",
        "",
        "# E01-R4 cache·numerical benchmark 보고",
        "",
        "## 판정",
        "",
    ]
    if Status == "READY_FOR_FULL_TRAINING":
        Lines.append(
            "strict 수치·cache·worker·runtime gate가 통과했다. 사용자에게 이 benchmark를 보고하기 전에는 장시간 3-seed 학습을 시작하지 않는다."
        )
    else:
        Lines.append(
            "고정 statistical workload를 줄이지 않은 보수적 projection이 24시간 gate를 넘거나 integrity gate가 실패하여 full training을 시작하지 않는다."
        )
    Lines.extend(
        [
            "",
            "## 불변 계약",
            "",
            f"- manifest SHA-256: `{Preflight['manifest']['manifest_sha256']}`",
            f"- manifest integrity rows: {Preflight['manifest']['manifest_total_row_count']:,}",
            f"- content-group crossing: {Preflight['manifest']['crossing_group_count']}",
            "- test rows: crossing용 두 field만 사용; statistics/predictions/metrics 0",
            "- workload: 32,768 samples/epoch × 20 epochs × 3 seeds",
            "- segment: fixed 8초, max 8/file, explicit valid mask",
            "- model input: waveform only; cache locator/source metadata는 feature가 아님",
            "",
            "## strict numerical gate",
            "",
            f"- unit/adversarial tests: {Tests['check_count']}/{Tests['check_count']} PASS",
            "- nested NaN JSON: recursive gate + `allow_nan=False` hard reject",
            "- masked NaN logits: finite loss여도 logits guard가 hard reject",
            "- precision: quality-first guarded FP32; CUDA autocast/GradScaler disabled",
            "- optimizer: FP32 gradient, parameters, step counter를 batch별 검사",
            f"- cached GPU pilot: {Pilot['guarded_batch_count']} guarded batches, finite loss {Pilot['final_loss']:.8f}, skip {Pilot['optimizer_skip_count']}",
            "",
            "## exact cache",
            "",
            f"- scope: non-test FMA/AIME {CacheGate['summary']['completed_entries']:,} entries",
            f"- estimated bytes: {StorageEstimate['estimated_total_cache_bytes']:,}",
            f"- actual NPY bytes: {CacheGate['summary']['cache_npy_bytes']:,} ({CacheGate['summary']['cache_npy_gib']:.3f} GiB)",
            f"- this-run build action: {CacheBuild['action']}",
            f"- build/verify seconds: {CacheBuild['build_and_verify_seconds']:.3f}",
            "- source locator/hash/sample-count pinned; reload max-abs-diff=0; raw originals retained",
            f"- AIME locator resolver: {AimeResolution['checked_record_count']}/{AimeResolution['checked_record_count']} ID assertions PASS, declared 1-based → resolved 0-based",
            f"- cache integrity: {CacheGate['status']}",
            "",
            "## Windows workers와 GPU",
            "",
            f"- worker 0/2/4 exact sequence: {WorkerBenchmark['status']}",
            f"- deterministic CPU threads: parent intra/inter {WorkerBenchmark['parent_intraop_threads']}/{WorkerBenchmark['parent_interop_threads']}, worker {WorkerBenchmark['worker_intraop_threads']}",
            f"- selected workers: {WorkerBenchmark['recommended_workers']} ({WorkerBenchmark['recommended_samples_per_second']:.3f} sample/s loader-only warm pass)",
            f"- GPU autotune batch: {GpuAutotune['recommended_batch_size']} ({GpuAutotune['recommended_segments_per_second']:.3f} guarded segment/s)",
            f"- realistic cached loader+GPU: {Pilot['end_to_end_samples_per_second']:.3f} sample/s, peak {Pilot['peak_allocated_gib']:.3f} GiB",
            "",
            "## 보수적 full-run projection",
            "",
            f"- measured rate: {Projection['measured_cached_end_to_end_samples_per_second']:.3f} sample/s",
            f"- safety factor: {Projection['projection_safety_factor']:.2f}",
            f"- conservative rate: {Projection['conservative_samples_per_second']:.3f} sample/s",
            f"- projected 3-seed wall time: {Projection['projected_three_seed_wall_hours']:.3f} hours",
            f"- performance-first gate: ≤{Projection['full_training_wall_gate_hours']:.1f} wall-hours",
            f"- status: {Projection['status']}",
            "- arbitrary 3 GPU-hour gate는 제거했고 statistical workload는 축소하지 않았다.",
            "",
            "## 실행 범위",
            "",
            "- full 3-seed training started: false",
            "- validation OOF/metric/checkpoint: 생성하지 않음",
            "- E02: 실행하지 않음",
            "- R1/R2/R3 code와 reports: 보존",
            "",
            "## 재현성",
            "",
            f"- source: `{RunManifest['source_code_directory']}`",
            f"- code inventory SHA-256: `{RunManifest['versions']['e01_r4_code_inventory_sha256']}`",
            f"- config SHA-256: `{RunManifest['versions']['config_sha256']}`",
            f"- git HEAD: `{RunManifest['versions']['git_head']}`",
            "",
        ]
    )
    return "\n".join(Lines)


def Main() -> int:
    Started = time.perf_counter()
    RunStartedUtc = datetime.now(timezone.utc).isoformat()
    DeepvoiceRoot = Path(__file__).resolve().parents[2]
    E01Root = Path(__file__).resolve().parent
    ConfigPath = E01Root / "config.json"
    Config = json.loads(ConfigPath.read_text(encoding="utf-8"))
    ThreadEvidence = ConfigureParentCpuThreads(Config)
    CacheRoot = DeepvoiceRoot / Config["cache_relative_path"]
    Config["resolved_cache_root"] = str(CacheRoot)
    ReportsRoot = DeepvoiceRoot / "reports"
    ArtifactsRoot = DeepvoiceRoot / "artifacts" / "e01_r4"
    Prefix = "e01-r4"
    Outputs = {
        "report": ReportsRoot / f"{Prefix}-experiment-batch.md",
        "preflight": ReportsRoot / f"{Prefix}-preflight.json",
        "tests": ReportsRoot / f"{Prefix}-unit-test-results.json",
        "storage": ReportsRoot / f"{Prefix}-cache-storage-gate.json",
        "aime_resolution": ReportsRoot / f"{Prefix}-aime-row-resolution.json",
        "cache": ReportsRoot / f"{Prefix}-cache-build.json",
        "cache_gate": ReportsRoot / f"{Prefix}-cache-integrity.json",
        "workers": ReportsRoot / f"{Prefix}-windows-workers.json",
        "autotune": ReportsRoot / f"{Prefix}-batch-autotune.json",
        "smoke": ReportsRoot / f"{Prefix}-tiny-gpu-smoke.json",
        "pilot": ReportsRoot / f"{Prefix}-cached-gpu-pilot.json",
        "projection": ReportsRoot / f"{Prefix}-runtime-projection.json",
        "run_manifest": ReportsRoot / f"{Prefix}-run-manifest.json",
        "code_inventory": ReportsRoot / f"{Prefix}-code-inventory.csv",
        "sampler_audit": ReportsRoot / f"{Prefix}-sampler-audit.csv",
        "artifact_run": ArtifactsRoot / "benchmark-run.json",
    }
    Collisions = [str(Value) for Value in Outputs.values() if Value.exists()]
    if Collisions:
        raise FileExistsError("Refusing to overwrite E01-R4 outputs: " + ", ".join(Collisions))

    Preflight = RunPreflight(DeepvoiceRoot, Config)
    if Preflight["status"] != "READY":
        raise RuntimeError(f"E01-R4 preflight blocked: {Preflight['blockers']}")
    Tests = RunAllTests(Config)
    TrainingRecords, ValidationRecords, ManifestSummary = LoadE01Records(
        DeepvoiceRoot / Config["manifest_relative_path"]
    )
    NonTestRecords = [*TrainingRecords, *ValidationRecords]
    AimeResolution = AuditAimeLocatorResolution(NonTestRecords)
    StorageEstimate = EstimateCacheStorage(
        NonTestRecords,
        CacheRoot,
        int(Config["sample_rate"]),
        12.0,
    )
    if StorageEstimate["status"] != "PASS":
        raise RuntimeError(f"E01-R4 cache disk gate blocked: {StorageEstimate}")
    CacheGateBefore = VerifyCompleteCache(NonTestRecords, CacheRoot)
    if CacheGateBefore["status"] == "PASS":
        ExistingSummary = CacheGateBefore["summary"]
        CacheBuild = {
            **ExistingSummary,
            "action": "REUSED_COMPLETE_NO_RAW_REDECODE",
            "build_and_verify_seconds": 0.0,
        }
    else:
        CacheBuild = BuildExactCache(
            NonTestRecords,
            CacheRoot,
            int(Config["sample_rate"]),
        )
        CacheBuild["action"] = "BUILT_OR_RESUMED_WITH_RAW_EXACT_VERIFICATION"
    CacheGate = VerifyCompleteCache(NonTestRecords, CacheRoot)
    if CacheGate["status"] != "PASS":
        raise RuntimeError(f"E01-R4 cache integrity blocked: {CacheGate}")

    Smoke = RunTinySmoke(Config, "cuda")
    GpuAutotune = BenchmarkGpuBatch(Config)
    if GpuAutotune["status"] != "PASS":
        raise RuntimeError(f"E01-R4 GPU autotune blocked: {GpuAutotune}")
    Config["batch_size"] = int(GpuAutotune["recommended_batch_size"])
    WorkerBenchmark = BenchmarkWindowsWorkers(TrainingRecords, Config)
    if WorkerBenchmark["status"] != "PASS":
        raise RuntimeError(f"E01-R4 worker benchmark blocked: {WorkerBenchmark}")
    Config["workers"] = int(WorkerBenchmark["recommended_workers"])
    Pilot = BenchmarkCachedEndToEndPilot(TrainingRecords, Config)
    Projection = ProjectRuntime(Config, Pilot, ValidationRecords, CacheGate)
    Status = Projection["status"]
    CodeInventory = BuildCodeInventory(E01Root)
    RunManifest = {
        "experiment_batch": Status,
        "experiment_id": "E01",
        "revision": "R4",
        "execution_phase": Config["execution_phase"],
        "scope": "cache and benchmark only before long full three-seed training",
        "source_code_directory": str(E01Root.relative_to(DeepvoiceRoot)).replace("\\", "/"),
        "lineage": {
            "prior_revisions_preserved": ["E01-R1", "E01-R2", "E01-R3"],
            "supersedes_resource_gate_evidence": ["E01-R1", "E01-R2", "E01-R3"],
        },
        "run_started_utc": RunStartedUtc,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - Started,
        "config": Config,
        "manifest_integrity": ManifestSummary,
        "test_usage_contract": {
            "allowed_fields": ["content_group_key", "recommended_content_split"],
            "test_statistics": 0,
            "test_predictions": 0,
            "test_metrics": 0,
        },
        "preflight_status": Preflight["status"],
        "unit_test_status": Tests["status"],
        "cache_storage_gate": StorageEstimate,
        "aime_row_resolution": AimeResolution,
        "cache_build": CacheBuild,
        "cache_integrity": CacheGate,
        "worker_benchmark": WorkerBenchmark,
        "tiny_gpu_smoke": Smoke,
        "gpu_autotune": GpuAutotune,
        "cached_gpu_pilot": Pilot,
        "runtime_projection": Projection,
        "full_training_started": False,
        "statistical_workload_reduced": False,
        "raw_sources_modified": False,
        "cpu_thread_determinism": ThreadEvidence,
        "versions": {
            "config_sha256": HashFile(ConfigPath),
            "e01_r4_code_inventory_sha256": InventoryDigest(CodeInventory),
            "e00_r2_contract_sha256": Config["e00_r2_contract_sha256"],
            "manifest_sha256": Config["manifest_sha256"],
            "git_head": RunCommand(["git", "rev-parse", "HEAD"], DeepvoiceRoot),
            "git_status_porcelain": RunCommand(["git", "status", "--short"], DeepvoiceRoot),
        },
    }
    AssertFinitePayload(RunManifest)
    Report = BuildReport(
        Status,
        Preflight,
        Tests,
        StorageEstimate,
        AimeResolution,
        CacheBuild,
        CacheGate,
        WorkerBenchmark,
        GpuAutotune,
        Pilot,
        Projection,
        RunManifest,
    )
    WriteBytesExclusive(Outputs["preflight"], JsonBytes(Preflight))
    WriteBytesExclusive(Outputs["tests"], JsonBytes(Tests))
    WriteBytesExclusive(Outputs["storage"], JsonBytes(StorageEstimate))
    WriteBytesExclusive(Outputs["aime_resolution"], JsonBytes(AimeResolution))
    WriteBytesExclusive(Outputs["cache"], JsonBytes(CacheBuild))
    WriteBytesExclusive(Outputs["cache_gate"], JsonBytes(CacheGate))
    WriteBytesExclusive(Outputs["workers"], JsonBytes(WorkerBenchmark))
    WriteBytesExclusive(Outputs["autotune"], JsonBytes(GpuAutotune))
    WriteBytesExclusive(Outputs["smoke"], JsonBytes(Smoke))
    WriteBytesExclusive(Outputs["pilot"], JsonBytes(Pilot))
    WriteBytesExclusive(Outputs["projection"], JsonBytes(Projection))
    WriteBytesExclusive(Outputs["run_manifest"], JsonBytes(RunManifest))
    WriteBytesExclusive(Outputs["code_inventory"], CsvBytes(CodeInventory))
    WriteBytesExclusive(Outputs["sampler_audit"], CsvBytes(SummarizeSampler(TrainingRecords)))
    WriteBytesExclusive(Outputs["artifact_run"], JsonBytes(RunManifest))
    WriteBytesExclusive(Outputs["report"], Report.encode("utf-8"))
    print(Report)
    return 0 if Status == "READY_FOR_FULL_TRAINING" else 3


if __name__ == "__main__":
    raise SystemExit(Main())
