# /// <summary>
# Reproducible E01 preflight, contract tests, measured resource gate and guarded full run
# /// </summary>

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .benchmark import (
    BenchmarkBalancedPilot,
    BenchmarkGpuBatch,
    BenchmarkLoaders,
    ProjectRuntime,
)
from .contract_adapter import HashFile
from .preflight import RunPreflight
from .records import LoadE01Records
from .run_tests import RunAllTests, RunTinySmoke
from .sampling import SummarizeSampler
from .train_e01 import RunFullTraining


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


def JsonBytes(Value: Any) -> bytes:
    return (
        json.dumps(Value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def WriteBytesExclusive(OutputPath: Path, Payload: bytes) -> None:
    if OutputPath.exists():
        raise FileExistsError(f"Refusing to overwrite E01 output: {OutputPath}")
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
        f"{Row['relative_path']}\0{Row['bytes']}\0{Row['sha256']}\n"
        for Row in Rows
    ).encode("utf-8")
    return hashlib.sha256(Payload).hexdigest()


def CsvBytes(Rows: list[dict[str, Any]]) -> bytes:
    if not Rows:
        raise ValueError("Cannot serialize an empty CSV")
    Lines: list[str] = []
    class MemoryWriter:
        def write(self, Text: str) -> int:
            Lines.append(Text)
            return len(Text)
    Writer = csv.DictWriter(MemoryWriter(), fieldnames=list(Rows[0]))
    Writer.writeheader()
    Writer.writerows(Rows)
    return "".join(Lines).encode("utf-8")


def BuildReport(
    Revision: str,
    Status: str,
    Preflight: dict[str, Any],
    Tests: dict[str, Any],
    Smoke: dict[str, Any],
    GpuBenchmark: dict[str, Any],
    LoaderBenchmark: dict[str, Any],
    Pilot: dict[str, Any],
    Projection: dict[str, Any],
    RunManifest: dict[str, Any],
    FullResult: dict[str, Any] | None,
) -> str:
    Lines = [
        f"EXPERIMENT_BATCH: {Status}",
        "",
        f"# E01-{Revision.upper()} log-mel CNN reference baseline 실행 보고",
        "",
        "## 판정",
        "",
    ]
    if Status == "COMPLETE":
        Lines.append(
            "고정 계약의 3-seed 전체 실행을 완료했다. 수치는 독립 E01 감사 전까지 후보 선택 근거로 사용하지 않는다."
        )
    else:
        Lines.append(
            "코드·실데이터 loader·GPU smoke는 통과했지만, 고정 workload의 실측 runtime projection이 3 GPU-hour gate를 넘어서 전체 3-seed 학습은 시작하지 않았다. 축소 seed·축소 split 결과를 E01 성능으로 주장하지 않는다."
        )
    Lines.extend(
        [
            "",
            "## 고정 계약",
            "",
            f"- manifest SHA-256: `{Preflight['manifest']['manifest_sha256']}`",
            f"- manifest integrity row count: {Preflight['manifest']['manifest_total_row_count']:,}",
            f"- split crossing group: {Preflight['manifest']['crossing_group_count']}",
            "- test isolation: crossing 검사용 `content_group_key`, `recommended_content_split`만 투영; test 통계 0건",
            "- segment: 8초, 파일당 최대 8개, 짧은 파일은 explicit valid-sample/frame mask를 정규화와 CNN pooling에 적용",
            "- feature: waveform만 사용; source/codec/rate/channel/path/provider metadata는 입력 금지",
            "- sampling: group-first speech/music 50:50, real/fake 균형, paired speech content, AIME provider 균형",
            "- seeds: 20260830 / 20260831 / 20260832",
            "",
            "## 환경 및 검증",
            "",
            f"- preflight: {Preflight['status']} ({len(Preflight['blockers'])} blockers)",
            f"- actual locator decode: {sum(Probe['decode_status'] == 'PASS' for Probe in Preflight['locator_probes'])}/4 PASS",
            f"- unit/contract tests: {Tests['check_count']}/{Tests['check_count']} PASS",
            f"- GPU tiny smoke: {Smoke['status']}, {Smoke['training_segments_per_second']:.3f} segment/s, peak {Smoke['peak_allocated_bytes'] / 1024**2:.2f} MiB",
            f"- GPU autotune: batch {GpuBenchmark['recommended_batch_size']}, {GpuBenchmark['recommended_segments_per_second']:.3f} segment/s, peak {GpuBenchmark['recommended_peak_allocated_bytes'] / 1024**3:.3f} GiB",
            f"- balanced real-locator pilot: {Pilot['sample_count']} samples, {Pilot['end_to_end_samples_per_second']:.3f} sample/s, peak {Pilot['peak_allocated_gib']:.3f} GiB",
            f"- balanced loader probe: {LoaderBenchmark['balanced_files_per_second']:.3f} file/s",
            "",
            "## 자원 판정",
            "",
            f"- workload: {Projection['samples_per_epoch']:,} samples/epoch × {RunManifest['config']['epochs']} epochs × 3 seeds = {Projection['training_samples_per_seed'] * 3:,} training decodes",
            f"- projected three-seed wall/GPU time: {Projection['projected_three_seed_wall_hours']:.3f} hours",
            f"- gate: {Projection['full_run_gate_gpu_hours']:.1f} GPU-hours and {Projection['full_run_gate_wall_hours']:.1f} wall-hours",
            f"- runtime gate status: {Projection['status']}",
            "- monetary cost: local existing hardware, incremental API/cloud cost KRW 0",
            "",
            "## 재현성",
            "",
            f"- config SHA-256: `{RunManifest['versions']['config_sha256']}`",
            f"- E01 code inventory SHA-256: `{RunManifest['versions']['e01_code_inventory_sha256']}`",
            f"- E00-R2 contract SHA-256: `{RunManifest['versions']['e00_r2_contract_sha256']}`",
            f"- git HEAD: `{RunManifest['versions']['git_head']}`",
            f"- run UTC: `{RunManifest['run_started_utc']}`",
            "",
            "## 산출물 범위",
            "",
        ]
    )
    if FullResult is None:
        Lines.extend(
            [
                "- 실제 validation prediction/OOF/metric/checkpoint는 생성하지 않았다.",
                "- tiny smoke와 pilot loss는 합성·소규모 실행 건전성 검사이며 E01 성능 결과가 아니다.",
                "- E02 학습은 수행하지 않았다.",
                "- R1 NaN 보고서와 R2 corrected-but-mutable-lineage 보고서는 보존하며, 독립 `experiments/e01_r3` snapshot인 이 revision이 resource gate evidence를 supersede한다.",
                "- R1 실행 뒤 동일 `experiments/e01` 경로가 동시 교정되어 R1 source byte 보존은 주장하지 않는다.",
            ]
        )
    else:
        Lines.extend(
            [
                f"- 3-seed RobustSelectionScore mean: {FullResult['score_mean']:.8f}",
                f"- population std: {FullResult['score_std_population']:.8f}",
                "- validation prediction, 7축 shortcut, content-group bootstrap, checkpoint를 seed별 보존했다.",
                "- E02 학습은 수행하지 않았다.",
            ]
        )
    Lines.extend(
        [
            "",
            "## 환경 재구성 명령",
            "",
            "```powershell",
            *Preflight["installation_commands_windows"],
            "```",
            "",
            "공식 설치 근거: https://pytorch.org/get-started/previous-versions/ , https://arrow.apache.org/docs/python/install.html",
            "",
        ]
    )
    return "\n".join(Lines)


def Main(Revision: str = "r3") -> int:
    Started = time.perf_counter()
    RunStartedUtc = datetime.now(timezone.utc).isoformat()
    DeepvoiceRoot = Path(__file__).resolve().parents[2]
    E01Root = Path(__file__).resolve().parent
    ConfigPath = E01Root / "config.json"
    Config = json.loads(ConfigPath.read_text(encoding="utf-8"))
    ReportsRoot = DeepvoiceRoot / "reports"
    ArtifactsRoot = DeepvoiceRoot / "artifacts" / "e01"
    Prefix = "e01" if Revision == "r1" else f"e01-{Revision}"
    ArtifactRunPath = (
        ArtifactsRoot / "run.json"
        if Revision == "r1"
        else ArtifactsRoot / Revision / "run.json"
    )
    Outputs = {
        "report": ReportsRoot / f"{Prefix}-experiment-batch.md",
        "preflight": ReportsRoot / f"{Prefix}-preflight.json",
        "tests": ReportsRoot / f"{Prefix}-unit-test-results.json",
        "smoke": ReportsRoot / f"{Prefix}-tiny-gpu-smoke.json",
        "autotune": ReportsRoot / f"{Prefix}-batch-autotune.json",
        "loader": ReportsRoot / f"{Prefix}-loader-benchmark.json",
        "pilot": ReportsRoot / f"{Prefix}-balanced-pilot.json",
        "projection": ReportsRoot / f"{Prefix}-runtime-projection.json",
        "run_manifest": ReportsRoot / f"{Prefix}-run-manifest.json",
        "code_inventory": ReportsRoot / f"{Prefix}-code-inventory.csv",
        "sampler_audit": ReportsRoot / f"{Prefix}-sampler-audit.csv",
        "full_result": ReportsRoot / f"{Prefix}-results.json",
        "artifact_run": ArtifactRunPath,
    }
    Collisions = [str(PathValue) for PathValue in Outputs.values() if PathValue.exists()]
    if Collisions:
        raise FileExistsError("Refusing to overwrite existing E01 outputs: " + ", ".join(Collisions))

    Preflight = RunPreflight(DeepvoiceRoot, Config)
    Tests = RunAllTests(Config)
    if Preflight["status"] != "READY":
        raise RuntimeError("Resource preflight failed: " + ", ".join(Preflight["blockers"]))
    Smoke = RunTinySmoke(Config, "cuda")
    TrainingRecords, ValidationRecords, ManifestSummary = LoadE01Records(
        DeepvoiceRoot / Config["manifest_relative_path"]
    )
    SamplerAudit = SummarizeSampler(TrainingRecords)
    GpuBenchmark = BenchmarkGpuBatch(Config)
    if GpuBenchmark["status"] != "PASS":
        raise RuntimeError(f"GPU autotune failed: {GpuBenchmark}")
    if int(Config["batch_size"]) > int(GpuBenchmark["recommended_batch_size"]):
        Config["batch_size"] = int(GpuBenchmark["recommended_batch_size"])
    LoaderBenchmark = BenchmarkLoaders(TrainingRecords, int(Config["sample_rate"]))
    Pilot = BenchmarkBalancedPilot(TrainingRecords, Config)
    if Pilot["status"] != "PASS":
        raise RuntimeError(f"Balanced pilot failed: {Pilot}")
    Projection = ProjectRuntime(
        Config,
        GpuBenchmark,
        LoaderBenchmark,
        ValidationRecords,
        Pilot,
    )
    FullResult = None
    if Projection["status"] == "READY":
        FullResult = RunFullTraining(DeepvoiceRoot, Config)
        Status = "COMPLETE"
    else:
        Status = "BLOCKED_RESOURCE"

    CodeInventory = BuildCodeInventory(E01Root)
    RunManifest = {
        "experiment_batch": Status,
        "experiment_id": "E01",
        "revision": Revision,
        "source_code_directory": str(E01Root.relative_to(DeepvoiceRoot)).replace("\\", "/"),
        "lineage": {
            "supersedes": (
                ["E01-R1", "E01-R2"]
                if Revision == "r3"
                else (["E01-R1"] if Revision == "r2" else [])
            ),
            "reason": (
                "R1 recorded nonfinite final_loss as PASS; R2 corrected numerics "
                "but shared a mutable experiments/e01 path. R3 is an independent "
                "experiments/e01_r3 snapshot with FP32 log-mel, GradScaler and a "
                "nonfinite hard guard."
                if Revision == "r3"
                else (
                    "R1 balanced pilot recorded nonfinite final_loss as PASS; "
                    "R2 uses FP32 log-mel, GradScaler and a nonfinite hard guard"
                    if Revision == "r2"
                    else None
                )
            ),
            "prior_report_outputs_preserved": Revision != "r1",
            "prior_source_code_byte_preserved": (
                False if Revision == "r3" else None
            ),
            "concurrent_mutation_limitation": (
                "The shared experiments/e01 directory changed after R1 before an "
                "independent snapshot was made; only R1 reports, not exact R1 source "
                "bytes, are preserved."
                if Revision == "r3"
                else None
            ),
        },
        "scope": "fixed log-mel CNN baseline only; no E02",
        "run_started_utc": RunStartedUtc,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - Started,
        "config": Config,
        "versions": {
            "config_sha256": HashFile(ConfigPath),
            "e01_code_inventory_sha256": InventoryDigest(CodeInventory),
            "e00_r2_contract_sha256": Config["e00_r2_contract_sha256"],
            "manifest_sha256": Config["manifest_sha256"],
            "git_head": RunCommand(["git", "rev-parse", "HEAD"], DeepvoiceRoot),
            "git_status_porcelain": RunCommand(
                ["git", "status", "--short"], DeepvoiceRoot
            ),
        },
        "manifest_integrity": ManifestSummary,
        "test_usage_contract": {
            "allowed_fields": [
                "content_group_key",
                "recommended_content_split",
            ],
            "test_statistics": 0,
            "test_predictions": 0,
            "test_metrics": 0,
        },
        "preflight_status": Preflight["status"],
        "unit_test_status": Tests["status"],
        "tiny_smoke_is_e01_performance_result": False,
        "balanced_pilot_is_e01_performance_result": False,
        "runtime_projection": Projection,
        "cost": {
            "hardware": "local GTX 1660 SUPER 6 GiB",
            "incremental_cloud_or_api_cost_krw": 0,
        },
        "full_training_started": FullResult is not None,
    }
    Report = BuildReport(
        Revision,
        Status,
        Preflight,
        Tests,
        Smoke,
        GpuBenchmark,
        LoaderBenchmark,
        Pilot,
        Projection,
        RunManifest,
        FullResult,
    )
    WriteBytesExclusive(Outputs["preflight"], JsonBytes(Preflight))
    WriteBytesExclusive(Outputs["tests"], JsonBytes(Tests))
    WriteBytesExclusive(Outputs["smoke"], JsonBytes(Smoke))
    WriteBytesExclusive(Outputs["autotune"], JsonBytes(GpuBenchmark))
    WriteBytesExclusive(Outputs["loader"], JsonBytes(LoaderBenchmark))
    WriteBytesExclusive(Outputs["pilot"], JsonBytes(Pilot))
    WriteBytesExclusive(Outputs["projection"], JsonBytes(Projection))
    WriteBytesExclusive(Outputs["run_manifest"], JsonBytes(RunManifest))
    WriteBytesExclusive(Outputs["code_inventory"], CsvBytes(CodeInventory))
    WriteBytesExclusive(Outputs["sampler_audit"], CsvBytes(SamplerAudit))
    if FullResult is not None:
        WriteBytesExclusive(Outputs["full_result"], JsonBytes(FullResult))
    WriteBytesExclusive(Outputs["artifact_run"], JsonBytes(RunManifest))
    WriteBytesExclusive(Outputs["report"], Report.encode("utf-8"))
    print(Report)
    return 0 if Status == "COMPLETE" else 3


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--revision", default="r3", choices=("r1", "r2", "r3"))
    Arguments = Parser.parse_args()
    raise SystemExit(Main(Arguments.revision))
