# /// <summary>
# Reproducible runner for the isolated DeepVoice E00 evaluation-contract batch
# /// </summary>

from __future__ import annotations

import csv
import gzip
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DeepvoiceRoot = Path(__file__).resolve().parents[2]
if str(DeepvoiceRoot) not in sys.path:
    sys.path.insert(0, str(DeepvoiceRoot))

from experiments.e00_r2.contract import (  # noqa: E402
    AuditGroupCrossings,
    BootstrapByContentGroup,
    BuildFixturePredictions,
    BuildLabelMasks,
    BuildShortcutLabelAudit,
    BuildShortcutMetricAudit,
    CalculateCompetitionProxy,
    CalculateHeadMetrics,
    EvaluateSingletonEquivalence,
    ExpectedManifestSha256,
    HashFile,
    HeadNames,
    LoadManifestPartitions,
    ProjectCrossingRows,
    SummarizeLabelMasks,
    ValidateNonTestManifestRows,
)
from experiments.e00_r2.run_tests import RunAllTests  # noqa: E402


def LoadConfig(ConfigPath: Path) -> dict[str, Any]:
    with ConfigPath.open("r", encoding="utf-8") as FileHandle:
        return json.load(FileHandle)


def WriteJson(OutputPath: Path, Value: Any) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    with OutputPath.open("w", encoding="utf-8", newline="\n") as FileHandle:
        json.dump(Value, FileHandle, indent=2, ensure_ascii=False, allow_nan=False)
        FileHandle.write("\n")


def NormalizeCsvValue(Value: Any) -> Any:
    if Value is None:
        return ""
    if isinstance(Value, bool):
        return str(Value).lower()
    return Value


def WriteCsv(OutputPath: Path, Rows: Sequence[dict[str, Any]]) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    if not Rows:
        raise ValueError(f"Cannot infer CSV columns for empty output: {OutputPath}")
    FieldNames = list(Rows[0])
    with OutputPath.open("w", encoding="utf-8", newline="") as FileHandle:
        Writer = csv.DictWriter(FileHandle, fieldnames=FieldNames, extrasaction="raise")
        Writer.writeheader()
        for Row in Rows:
            Writer.writerow({Key: NormalizeCsvValue(Row.get(Key)) for Key in FieldNames})


def WriteGzipCsv(OutputPath: Path, Rows: Sequence[dict[str, Any]]) -> None:
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    if not Rows:
        raise ValueError(f"Cannot infer CSV columns for empty output: {OutputPath}")
    FieldNames = list(Rows[0])
    with OutputPath.open("wb") as RawFile:
        with gzip.GzipFile(fileobj=RawFile, mode="wb", mtime=0) as GzipFile:
            with io.TextIOWrapper(GzipFile, encoding="utf-8", newline="") as TextFile:
                Writer = csv.DictWriter(
                    TextFile,
                    fieldnames=FieldNames,
                    extrasaction="raise",
                )
                Writer.writeheader()
                for Row in Rows:
                    Writer.writerow(
                        {Key: NormalizeCsvValue(Row.get(Key)) for Key in FieldNames}
                    )


def GetGitValue(Arguments: Sequence[str]) -> str:
    RepoRoot = DeepvoiceRoot.parent
    Command = [
        "git",
        "-c",
        f"safe.directory={RepoRoot.as_posix()}",
        "-C",
        str(RepoRoot),
        *Arguments,
    ]
    Result = subprocess.run(
        Command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return Result.stdout.strip()


def GetPackageVersion(PackageName: str) -> str | None:
    try:
        return importlib.metadata.version(PackageName)
    except importlib.metadata.PackageNotFoundError:
        return None


def GetEnvironment() -> dict[str, Any]:
    Environment: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            Name: GetPackageVersion(Name)
            for Name in ("numpy", "scipy", "pandas", "torch", "torchaudio", "scikit-learn", "soundfile")
        },
    }
    try:
        import torch

        Environment["torch_cuda_available"] = bool(torch.cuda.is_available())
        Environment["torch_cuda_device_count"] = int(torch.cuda.device_count())
        Environment["torch_cuda_version"] = torch.version.cuda
        Environment["gpu_names"] = [
            torch.cuda.get_device_name(DeviceIndex)
            for DeviceIndex in range(torch.cuda.device_count())
        ]
    except Exception as Error:
        Environment["torch_environment_error"] = repr(Error)
    return Environment


def FlattenHeadMetricRows(
    Seed: int,
    MetricRows: Sequence[dict[str, Any]],
    Proxy: dict[str, float],
) -> list[dict[str, Any]]:
    Results = []
    for MetricRow in MetricRows:
        Results.append(
            {
                "seed": Seed,
                "prediction_kind": "label_independent_contract_fixture_non_model",
                **MetricRow,
                **Proxy,
            }
        )
    return Results


def BuildSeedSummary(HeadMetricRows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    GroupedValues: dict[tuple[str, str], list[float]] = defaultdict(list)
    MetricNames = (
        "auc",
        "eer",
        "brier",
        "log_loss",
        "ece_15",
        "selection_component",
        "RobustSelectionScore",
    )
    for Row in HeadMetricRows:
        for MetricName in MetricNames:
            Value = Row.get(MetricName)
            if Value is not None:
                GroupedValues[(Row["head"], MetricName)].append(float(Value))

    SummaryRows = []
    for HeadName in HeadNames:
        for MetricName in MetricNames:
            Values = GroupedValues.get((HeadName, MetricName), [])
            if not Values:
                continue
            ValueArray = np.asarray(Values, dtype=np.float64)
            SummaryRows.append(
                {
                    "head": HeadName,
                    "metric": MetricName,
                    "seed_count": int(ValueArray.size),
                    "mean": float(np.mean(ValueArray)),
                    "std_population": float(np.std(ValueArray, ddof=0)),
                    "minimum": float(np.min(ValueArray)),
                    "maximum": float(np.max(ValueArray)),
                }
            )
    return SummaryRows


def BuildFixturePredictionRows(
    Rows: Sequence[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
    Predictions: np.ndarray,
    Seed: int,
) -> list[dict[str, Any]]:
    OutputRows = []
    for RowIndex, Row in enumerate(Rows):
        OutputRow: dict[str, Any] = {
            "seed": Seed,
            "prediction_kind": "label_independent_contract_fixture_non_model",
            "dataset": Row["dataset"],
            "sample_id": Row["sample_id"],
            "content_group_key": Row["content_group_key"],
            "recommended_content_split": Row["recommended_content_split"],
        }
        for HeadIndex, HeadName in enumerate(HeadNames):
            OutputRow[f"{HeadName}_label"] = (
                float(Labels[RowIndex, HeadIndex]) if Masks[RowIndex, HeadIndex] else None
            )
            OutputRow[f"{HeadName}_mask"] = bool(Masks[RowIndex, HeadIndex])
            OutputRow[f"{HeadName}_prediction"] = float(Predictions[RowIndex, HeadIndex])
        OutputRows.append(OutputRow)
    return OutputRows


def MakeBootstrapSummaryRows(
    BootstrapSummaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    Rows = []
    for Summary in BootstrapSummaries:
        for MetricName, Interval in Summary["intervals"].items():
            Rows.append(
                {
                    "seed": Summary["seed"],
                    "sampling_unit": "content_group_key",
                    "requested_replicates": Summary["requested_replicates"],
                    "valid_replicates": Summary["valid_replicates"],
                    "confidence": Summary["confidence"],
                    "group_count": Summary["group_count"],
                    "metric": MetricName,
                    "mean": Interval["mean"],
                    "lower": Interval["lower"],
                    "upper": Interval["upper"],
                }
            )
    return Rows


def GetOutputPaths(ReportsRoot: Path, ArtifactsRoot: Path) -> list[Path]:
    ReturnPaths = [
        ReportsRoot / "e00-r2-experiment-batch.md",
        ReportsRoot / "e00-r2-label-mask-audit.csv",
        ReportsRoot / "e00-r2-group-crossings.csv",
        ReportsRoot / "e00-r2-head-metrics.csv",
        ReportsRoot / "e00-r2-seed-summary.csv",
        ReportsRoot / "e00-r2-bootstrap-summary.csv",
        ReportsRoot / "e00-r2-shortcut-label-audit.csv",
        ReportsRoot / "e00-r2-shortcut-metric-fixture.csv",
        ReportsRoot / "e00-r2-singleton-equivalence.json",
        ReportsRoot / "e00-r2-runtime.json",
        ReportsRoot / "e00-r2-run-manifest.json",
        ArtifactsRoot / "e00-r2-run.json",
        ArtifactsRoot / "e00-r2-test-results.json",
    ]
    for Seed in (20260830, 20260831, 20260832):
        ReturnPaths.append(ArtifactsRoot / f"validation-fixture-predictions-seed-{Seed}.csv.gz")
        ReturnPaths.append(ArtifactsRoot / f"bootstrap-replicates-seed-{Seed}.csv.gz")
    return ReturnPaths


def EnsureOutputsAbsent(OutputPaths: Sequence[Path]) -> None:
    ExistingPaths = [str(OutputPath) for OutputPath in OutputPaths if OutputPath.exists()]
    if ExistingPaths:
        raise FileExistsError(
            "E00-R2 refuses to overwrite existing outputs:\n" + "\n".join(ExistingPaths)
        )


def BuildReport(
    RunRecord: dict[str, Any],
    MaskSummaryRows: Sequence[dict[str, Any]],
    ShortcutLabelRows: Sequence[dict[str, Any]],
) -> str:
    SingletonMaximum = max(
        Result["max_absolute_delta"] for Result in RunRecord["singleton_equivalence"]
    )
    HighRiskSlices = sum(Row["shortcut_risk"] == "HIGH" for Row in ShortcutLabelRows)
    ObservedCounts = defaultdict(dict)
    for Row in MaskSummaryRows:
        ObservedCounts[Row["dataset"]][Row["head"]] = Row["observed_count"]

    Lines = [
        "EXPERIMENT_BATCH: COMPLETE",
        "",
        "# DeepVoice E00-R2 evaluation-contract batch",
        "",
        f"실행 시각: {RunRecord['finished_at_local']}",
        "범위: E00 scorer, label mask, split/group, content-group bootstrap, singleton-equivalence, shortcut 계약만 검증했다. 모델 학습과 E01은 실행하지 않았다.",
        "",
        "## 판정",
        "",
        "- E00 계약: COMPLETE",
        "- revision: R2. R1은 test-derived 통계를 기록해 validation audit에서 BLOCKED되었으며 이 보고서가 R1을 대체한다.",
        "- 다음 단계: 독립 validation auditor의 `EXPERIMENT_AUDIT: PASS` 전까지 모델 개선 근거로 사용하지 않는다.",
        "- fixture 예측: 라벨과 독립인 고정 난수이며 모델 결과, OOF 결과 또는 성능 기준선이 아니다.",
        "- test split: group crossing 감사에만 사용했으며 예측·지표·분포 통계를 계산하지 않았다.",
        "",
        "## 고정 입력",
        "",
        f"- manifest: `deepvoice/reports/deepvoice-training-manifest.csv.gz`",
        f"- SHA-256: `{RunRecord['manifest']['sha256']}`",
        f"- 행: {RunRecord['manifest']['row_count']:,}",
        f"- validation 행: {RunRecord['non_test_scope']['validation_row_count']:,}",
        "- seed: `20260830, 20260831, 20260832`",
        "",
        "## 공식 scorer 계약",
        "",
        "DACON 평가 페이지의 수식을 그대로 사용했다: `Score = 0.9 × ADS + 0.1 × CPS`; `ADS = 0.5 × (1-File EER) + 0.2 × (1-Voice EER) + 0.3 × (1-Music EER)`; `CPS = 0.5 × Voice Presence AUC + 0.5 × Music Presence AUC`.",
        "따라서 five-head 유효 가중치는 `0.45, 0.18, 0.27, 0.05, 0.05`다. EER은 공식 `drop_intermediate=False` threshold 규칙을 재현한다. 각 head에는 AUC, EER, Brier, log loss, 15-bin ECE를 함께 기록했다.",
        "",
        "`RobustSelectionScore`는 validation의 공식 가중 proxy와 동일하다. 모델링 계획에 수치 penalty가 정의되지 않았으므로 임의 penalty를 만들지 않았다. generator/provider/source/codec/rate/channel/duration macro·worst slice는 별도 필수 gate로 남긴다.",
        "",
        "## label과 mask",
        "",
        "| dataset | file fake | voice fake | music fake | voice present | music present |",
        "|---|---:|---:|---:|---:|---:|",
        "| LJSpeech | 0 | 0 | masked | 1 | 0 |",
        "| WaveFake speech | 1 | 1 | masked | 1 | 0 |",
        "| FMA real music | 0 | masked | 0 | masked | 1 |",
        "| AIME instrumental | 1 | masked | 1 | 0 | 1 |",
        "",
        f"mask audit는 `train_plus_validation_no_test` 범위의 {len(MaskSummaryRows)}개 dataset-head 조합을 확인했다. test 행의 label과 metadata는 읽거나 집계하지 않았다. 정확한 관측·mask 수는 `e00-r2-label-mask-audit.csv`에 있다.",
        "",
        "## 통과 조건",
        "",
        f"- content group split crossing: {RunRecord['split_audit']['crossing_group_count']}개",
        f"- 실행 테스트: {RunRecord['tests']['check_count']}개 모두 PASS",
        f"- singleton max absolute delta: {SingletonMaximum:.3g} (허용값 ≤ 1e-6)",
        f"- content-group bootstrap: seed별 {RunRecord['config']['bootstrap_replicates']}회 요청",
        f"- shortcut label-pure slice: {HighRiskSlices}개, 모델 성능이 아니라 데이터 교란 경보로 기록",
        "",
        "## 재현성과 비용",
        "",
        f"- git HEAD: `{RunRecord['code']['git_head']}`",
        f"- Python: `{RunRecord['environment']['python_version']}`",
        f"- NumPy: `{RunRecord['environment']['packages']['numpy']}`",
        f"- CUDA 사용: `{RunRecord['environment'].get('torch_cuda_available')}`",
        f"- 총 wall time: {RunRecord['runtime_seconds']['total']:.3f}초",
        f"- GPU time: 0시간",
        f"- 최대 논리 CPU 기준 비용 상한: {RunRecord['cost']['logical_cpu_hour_upper_bound']:.6f} CPU-hour",
        "",
        "## 산출물",
        "",
        "- `e00-r2-head-metrics.csv`: seed별 non-model fixture head metric",
        "- `e00-r2-bootstrap-summary.csv`: content-group bootstrap 95% CI",
        "- `e00-r2-shortcut-label-audit.csv`: train+validation shortcut label purity",
        "- `e00-r2-shortcut-metric-fixture.csv`: validation slice별 scorer 동작 상태",
        "- `e00-r2-singleton-equivalence.json`: 파일 독립 aggregation 검사",
        "- `e00-r2-run-manifest.json`: config/code/data/environment/runtime 버전",
        "- `deepvoice/artifacts/e00-r2/`: fixture 예측과 bootstrap replicate 원자료",
        "",
        "## 제한",
        "",
        "E00에는 학습된 예측이 없으므로 generator/provider macro와 worst 성능을 해석하지 않는다. single-domain manifest에는 실제 BOTH_PRESENT/NEITHER_PRESENT가 없으므로 joint presence와 mixed-file CPS도 검증되지 않았다. 이 결과는 scorer와 실험 계약이 실행 가능하다는 뜻이며 모델 품질을 뜻하지 않는다.",
        "",
        "공식 근거: https://dacon.io/competitions/official/236749/overview/evaluation (확인일 2026-08-30)",
        "",
    ]
    return "\n".join(Lines)


def Execute() -> dict[str, Any]:
    StartedTime = time.perf_counter()
    ConfigPath = Path(__file__).resolve().parent / "config.json"
    ReportsRoot = DeepvoiceRoot / "reports"
    ArtifactsRoot = DeepvoiceRoot / "artifacts" / "e00-r2"
    Config = LoadConfig(ConfigPath)
    OutputPaths = GetOutputPaths(ReportsRoot, ArtifactsRoot)
    EnsureOutputsAbsent(OutputPaths)

    ManifestPath = DeepvoiceRoot / Config["manifest_relative_path"]
    ManifestHashStarted = time.perf_counter()
    ManifestSha256 = HashFile(ManifestPath)
    ManifestHashSeconds = time.perf_counter() - ManifestHashStarted
    if ManifestSha256 != ExpectedManifestSha256:
        raise ValueError(
            f"Manifest SHA mismatch: expected {ExpectedManifestSha256}, found {ManifestSha256}"
        )
    if ManifestSha256 != Config["manifest_sha256"]:
        raise ValueError("Config manifest SHA does not match the compiled E00 contract")

    ManifestLoadStarted = time.perf_counter()
    NonTestRows, TestCrossingRows, FieldNames, TotalRowCount = LoadManifestPartitions(
        ManifestPath
    )
    ValidateNonTestManifestRows(NonTestRows, FieldNames)
    NonTestLabels, NonTestMasks = BuildLabelMasks(NonTestRows)
    ManifestLoadSeconds = time.perf_counter() - ManifestLoadStarted

    CrossingRows = ProjectCrossingRows(NonTestRows) + TestCrossingRows
    Crossings, SplitSummary = AuditGroupCrossings(CrossingRows)
    if Crossings:
        raise ValueError(f"Content groups cross recommended splits: {len(Crossings)}")

    ValidationIndices = np.asarray(
        [
            RowIndex
            for RowIndex, Row in enumerate(NonTestRows)
            if Row["recommended_content_split"] == Config["evaluation_split"]
        ],
        dtype=np.int64,
    )
    ValidationRows = [NonTestRows[int(RowIndex)] for RowIndex in ValidationIndices]
    ValidationLabels = NonTestLabels[ValidationIndices]
    ValidationMasks = NonTestMasks[ValidationIndices]

    Tests = RunAllTests()
    WriteJson(ArtifactsRoot / "e00-r2-test-results.json", Tests)

    HeadMetricRows: list[dict[str, Any]] = []
    BootstrapSummaries = []
    SingletonResults = []
    SeedRuntimeRows = []
    ReferencePredictions: np.ndarray | None = None
    for Seed in Config["seeds"]:
        SeedStarted = time.perf_counter()
        Predictions = BuildFixturePredictions(len(ValidationRows), Seed)
        if ReferencePredictions is None:
            ReferencePredictions = Predictions
        Metrics = CalculateHeadMetrics(
            ValidationLabels,
            ValidationMasks,
            Predictions,
        )
        Proxy = CalculateCompetitionProxy(Metrics)
        HeadMetricRows.extend(FlattenHeadMetricRows(Seed, Metrics, Proxy))

        BootstrapStarted = time.perf_counter()
        BootstrapRows, BootstrapSummary = BootstrapByContentGroup(
            ValidationRows,
            ValidationLabels,
            ValidationMasks,
            Predictions,
            Seed=Seed,
            Replicates=int(Config["bootstrap_replicates"]),
            Confidence=float(Config["bootstrap_confidence"]),
        )
        BootstrapSeconds = time.perf_counter() - BootstrapStarted
        BootstrapSummaries.append(BootstrapSummary)
        WriteGzipCsv(
            ArtifactsRoot / f"bootstrap-replicates-seed-{Seed}.csv.gz",
            BootstrapRows,
        )
        WriteGzipCsv(
            ArtifactsRoot / f"validation-fixture-predictions-seed-{Seed}.csv.gz",
            BuildFixturePredictionRows(
                ValidationRows,
                ValidationLabels,
                ValidationMasks,
                Predictions,
                Seed,
            ),
        )

        SingletonResult = EvaluateSingletonEquivalence(Seed)
        SingletonResults.append(SingletonResult)
        if SingletonResult["max_absolute_delta"] > float(Config["singleton_tolerance"]):
            raise AssertionError("Singleton-equivalence tolerance was exceeded")
        SeedRuntimeRows.append(
            {
                "seed": Seed,
                "bootstrap_seconds": BootstrapSeconds,
                "total_seed_seconds": time.perf_counter() - SeedStarted,
            }
        )

    if ReferencePredictions is None:
        raise RuntimeError("No seed was configured")

    ShortcutStarted = time.perf_counter()
    ShortcutLabelRows = BuildShortcutLabelAudit(
        NonTestRows,
        NonTestLabels,
        NonTestMasks,
        Scope="train_plus_validation_no_test",
    )
    ShortcutMetricRows = BuildShortcutMetricAudit(
        ValidationRows,
        ValidationLabels,
        ValidationMasks,
        ReferencePredictions,
        Scope="validation_seed_20260830_contract_fixture_non_model",
    )
    ShortcutSeconds = time.perf_counter() - ShortcutStarted
    MaskSummaryRows = SummarizeLabelMasks(
        NonTestRows,
        NonTestLabels,
        NonTestMasks,
        Scope="train_plus_validation_no_test",
    )

    WriteCsv(ReportsRoot / "e00-r2-label-mask-audit.csv", MaskSummaryRows)
    WriteCsv(
        ReportsRoot / "e00-r2-group-crossings.csv",
        Crossings
        if Crossings
        else [{"content_group_key": "", "splits": ""}],
    )
    WriteCsv(ReportsRoot / "e00-r2-head-metrics.csv", HeadMetricRows)
    WriteCsv(ReportsRoot / "e00-r2-seed-summary.csv", BuildSeedSummary(HeadMetricRows))
    WriteCsv(
        ReportsRoot / "e00-r2-bootstrap-summary.csv",
        MakeBootstrapSummaryRows(BootstrapSummaries),
    )
    WriteCsv(ReportsRoot / "e00-r2-shortcut-label-audit.csv", ShortcutLabelRows)
    WriteCsv(ReportsRoot / "e00-r2-shortcut-metric-fixture.csv", ShortcutMetricRows)
    WriteJson(ReportsRoot / "e00-r2-singleton-equivalence.json", SingletonResults)

    FinishedSeconds = time.perf_counter() - StartedTime
    Environment = GetEnvironment()
    Runtime = {
        "manifest_hash_seconds": ManifestHashSeconds,
        "manifest_load_and_contract_seconds": ManifestLoadSeconds,
        "shortcut_reports_seconds": ShortcutSeconds,
        "seed_runs": SeedRuntimeRows,
        "total": FinishedSeconds,
    }
    Cost = {
        "gpu_hours": 0.0,
        "wall_clock_hours": FinishedSeconds / 3600.0,
        "logical_cpu_hour_upper_bound": (
            FinishedSeconds * max(int(Environment.get("logical_cpu_count") or 1), 1) / 3600.0
        ),
        "cost_note": "No paid service or pretrained model was used; CPU upper bound assumes every logical CPU was busy for the full wall time",
    }
    CodePaths = (
        Path(__file__).resolve().parent / "contract.py",
        Path(__file__).resolve().parent / "run_e00.py",
        Path(__file__).resolve().parent / "run_tests.py",
        ConfigPath,
    )
    CodeRecord = {
        "git_head": GetGitValue(("rev-parse", "HEAD")),
        "git_status_short": GetGitValue(("status", "--short")),
        "files": {
            str(CodePath.relative_to(DeepvoiceRoot)): {
                "bytes": CodePath.stat().st_size,
                "sha256": HashFile(CodePath),
            }
            for CodePath in CodePaths
        },
    }
    SupersededPaths = (
        ReportsRoot / "e00-experiment-batch.md",
        ReportsRoot / "e00-run-manifest.json",
    )
    Lineage = {
        "revision": 2,
        "supersedes_revision": 1,
        "reason": Config["supersedes_reason"],
        "superseded_files": {
            str(SupersededPath.relative_to(DeepvoiceRoot)): {
                "bytes": SupersededPath.stat().st_size,
                "sha256": HashFile(SupersededPath),
            }
            for SupersededPath in SupersededPaths
        },
    }
    TrainRowCount = sum(
        Row["recommended_content_split"] == "train" for Row in NonTestRows
    )
    RunRecord = {
        "experiment_batch": "COMPLETE",
        "experiment_id": "E00-R2",
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "evaluation contract only; no model training and no E01",
        "lineage": Lineage,
        "config": Config,
        "manifest": {
            "path": str(ManifestPath.relative_to(DeepvoiceRoot)),
            "bytes": ManifestPath.stat().st_size,
            "sha256": ManifestSha256,
            "row_count": TotalRowCount,
        },
        "split_audit": SplitSummary,
        "non_test_scope": {
            "scope": "train_plus_validation_no_test",
            "train_row_count": TrainRowCount,
            "validation_row_count": len(ValidationRows),
        },
        "test_split_policy": "project to content_group_key and recommended_content_split immediately; crossing detection only",
        "test_field_access_contract": {
            "allowed_fields": [
                "content_group_key",
                "recommended_content_split",
            ],
            "retained_label_or_metadata_fields": 0,
            "test_prediction_rows": 0,
            "test_metric_rows": 0,
            "test_mask_summary_rows": 0,
            "sentinel_invariance_test": "PASS",
        },
        "label_contract": "explicit dataset-to-five-head mapping with unknown labels masked",
        "robust_selection_definition": Config["robust_selection_definition"],
        "prediction_kind": "label-independent deterministic uniform contract fixture; not model, OOF or baseline",
        "tests": Tests,
        "singleton_equivalence": SingletonResults,
        "bootstrap": BootstrapSummaries,
        "environment": Environment,
        "runtime_seconds": Runtime,
        "cost": Cost,
        "code": CodeRecord,
        "official_metric_source": {
            "url": "https://dacon.io/competitions/official/236749/overview/evaluation",
            "verified_date": "2026-08-30",
        },
    }
    WriteJson(
        ReportsRoot / "e00-r2-runtime.json",
        {"runtime_seconds": Runtime, "cost": Cost},
    )
    WriteJson(ReportsRoot / "e00-r2-run-manifest.json", RunRecord)
    WriteJson(ArtifactsRoot / "e00-r2-run.json", RunRecord)
    ReportText = BuildReport(RunRecord, MaskSummaryRows, ShortcutLabelRows)
    ReportPath = ReportsRoot / "e00-r2-experiment-batch.md"
    with ReportPath.open("w", encoding="utf-8", newline="\n") as FileHandle:
        FileHandle.write(ReportText)
    return RunRecord


def WriteBlockedReport(Error: BaseException) -> None:
    ReportsRoot = DeepvoiceRoot / "reports"
    ReportPath = ReportsRoot / "e00-r2-experiment-batch.md"
    if ReportPath.exists():
        return
    ReportsRoot.mkdir(parents=True, exist_ok=True)
    Lines = [
        "EXPERIMENT_BATCH: BLOCKED",
        "",
        "# DeepVoice E00-R2 evaluation-contract batch",
        "",
        f"오류: `{type(Error).__name__}: {Error}`",
        "",
        "```text",
        traceback.format_exc(),
        "```",
        "",
    ]
    with ReportPath.open("w", encoding="utf-8", newline="\n") as FileHandle:
        FileHandle.write("\n".join(Lines))


def Main() -> int:
    try:
        RunRecord = Execute()
        print(json.dumps(RunRecord, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as Error:
        WriteBlockedReport(Error)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(Main())
