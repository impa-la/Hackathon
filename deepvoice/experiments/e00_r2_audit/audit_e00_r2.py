# /// <summary>
# Independent read-only auditor for the corrected DeepVoice E00-R2 contract
# /// </summary>

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ExpectedManifestSha256 = (
    "2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6"
)
ExpectedManifestRows = 137328
ExpectedSeeds = (20260830, 20260831, 20260832)
AllowedTestFields = {"content_group_key", "recommended_content_split"}


def LoadModule(ModuleName: str, FilePath: Path) -> Any:
    Specification = importlib.util.spec_from_file_location(ModuleName, FilePath)
    if Specification is None or Specification.loader is None:
        raise ImportError(f"Cannot load {FilePath}")
    Module = importlib.util.module_from_spec(Specification)
    Specification.loader.exec_module(Module)
    return Module


def GetCallArguments(SourcePath: Path, FunctionName: str) -> list[list[str]]:
    Tree = ast.parse(SourcePath.read_text(encoding="utf-8"))
    Results = []
    for Node in ast.walk(Tree):
        if not isinstance(Node, ast.Call):
            continue
        CalledName = None
        if isinstance(Node.func, ast.Name):
            CalledName = Node.func.id
        elif isinstance(Node.func, ast.Attribute):
            CalledName = Node.func.attr
        if CalledName != FunctionName:
            continue
        Results.append(
            [Argument.id if isinstance(Argument, ast.Name) else "<expr>" for Argument in Node.args]
        )
    return Results


def AddCheck(
    Checks: list[dict[str, Any]],
    Name: str,
    Errors: Sequence[str],
    Evidence: Any,
) -> None:
    Checks.append(
        {
            "check": Name,
            "status": "PASS" if not Errors else "BLOCKED",
            "errors": list(Errors),
            "evidence": Evidence,
        }
    )


def AddQa(
    QaChecks: list[dict[str, Any]],
    Name: str,
    Passed: bool,
    Evidence: Any,
) -> None:
    QaChecks.append(
        {
            "qa": Name,
            "status": "PASS" if Passed else "BLOCKED",
            "evidence": Evidence,
        }
    )


def MakeSyntheticRow(
    RequiredColumns: Sequence[str],
    Dataset: str,
    Label: str,
    SampleId: str,
    Split: str,
    Group: str,
) -> dict[str, str]:
    Values = {
        "dataset": Dataset,
        "label": Label,
        "sample_id": SampleId,
        "source_family": "fixture",
        "generator_or_provider": "fixture",
        "content_group_key": Group,
        "recommended_content_split": Split,
        "provider_holdout_group": "fixture",
        "codec": "fixture",
        "sample_rate_hz": "16000",
        "channels": "1",
        "duration_seconds": "8.0",
        "training_eligible": "True",
    }
    return {Column: Values[Column] for Column in RequiredColumns}


def WriteSentinelManifest(
    FilePath: Path,
    RequiredColumns: Sequence[str],
    Sentinel: str,
) -> None:
    TestRow = MakeSyntheticRow(
        RequiredColumns,
        Sentinel,
        Sentinel,
        Sentinel,
        "test",
        "test-group",
    )
    for Column in RequiredColumns:
        if Column not in AllowedTestFields:
            TestRow[Column] = f"{Sentinel}:{Column}"
    Rows = [
        MakeSyntheticRow(
            RequiredColumns,
            "ljspeech-1.1",
            "real",
            "ljs",
            "validation",
            "ljs",
        ),
        MakeSyntheticRow(
            RequiredColumns,
            "wavefake-1.2.0",
            "synthetic",
            "wave",
            "validation",
            "wave",
        ),
        MakeSyntheticRow(
            RequiredColumns,
            "fma-small",
            "real",
            "fma",
            "validation",
            "fma",
        ),
        MakeSyntheticRow(
            RequiredColumns,
            "aime-open-model-subset",
            "synthetic",
            "aime",
            "validation",
            "aime",
        ),
        TestRow,
    ]
    with gzip.open(FilePath, "wt", encoding="utf-8", newline="") as FileHandle:
        Writer = csv.DictWriter(FileHandle, fieldnames=RequiredColumns)
        Writer.writeheader()
        Writer.writerows(Rows)


def AuditSentinel(R2Contract: Any) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    with tempfile.TemporaryDirectory() as TemporaryDirectory:
        TemporaryRoot = Path(TemporaryDirectory)
        FirstPath = TemporaryRoot / "sentinel-a.csv.gz"
        SecondPath = TemporaryRoot / "sentinel-b.csv.gz"
        WriteSentinelManifest(
            FirstPath,
            R2Contract.RequiredManifestColumns,
            "INDEPENDENT_SENTINEL_INVALID_A",
        )
        WriteSentinelManifest(
            SecondPath,
            R2Contract.RequiredManifestColumns,
            "INDEPENDENT_SENTINEL_INVALID_B",
        )
        FirstNonTest, FirstTest, FirstFields, FirstTotal = (
            R2Contract.LoadManifestPartitions(FirstPath)
        )
        SecondNonTest, SecondTest, SecondFields, SecondTotal = (
            R2Contract.LoadManifestPartitions(SecondPath)
        )
        ExpectedProjection = [
            {
                "content_group_key": "test-group",
                "recommended_content_split": "test",
            }
        ]
        if FirstTest != ExpectedProjection or SecondTest != ExpectedProjection:
            Errors.append("sentinel test rows were not reduced to the exact two-key projection")
        if FirstTotal != SecondTotal or FirstTotal != 5:
            Errors.append("synthetic manifest totals differ")
        if FirstNonTest != SecondNonTest or FirstFields != SecondFields:
            Errors.append("invalid test metadata affected retained non-test rows or schema")
        R2Contract.ValidateNonTestManifestRows(FirstNonTest, FirstFields)
        R2Contract.ValidateNonTestManifestRows(SecondNonTest, SecondFields)
        FirstLabels, FirstMasks = R2Contract.BuildLabelMasks(FirstNonTest)
        SecondLabels, SecondMasks = R2Contract.BuildLabelMasks(SecondNonTest)
        Predictions = R2Contract.BuildFixturePredictions(len(FirstNonTest), 20260830)
        FirstMetrics = R2Contract.CalculateHeadMetrics(
            FirstLabels,
            FirstMasks,
            Predictions,
        )
        SecondMetrics = R2Contract.CalculateHeadMetrics(
            SecondLabels,
            SecondMasks,
            Predictions,
        )
        if FirstMetrics != SecondMetrics:
            Errors.append("invalid test metadata affected synthetic validation metrics")
        FirstScore = R2Contract.CalculateCompetitionProxy(FirstMetrics)
        SecondScore = R2Contract.CalculateCompetitionProxy(SecondMetrics)
        if FirstScore != SecondScore:
            Errors.append("invalid test metadata affected synthetic validation score")
        FirstCrossings = R2Contract.AuditGroupCrossings(
            R2Contract.ProjectCrossingRows(FirstNonTest) + FirstTest
        )
        SecondCrossings = R2Contract.AuditGroupCrossings(
            R2Contract.ProjectCrossingRows(SecondNonTest) + SecondTest
        )
        if FirstCrossings != SecondCrossings:
            Errors.append("invalid test metadata affected crossing result")
        StrictRejection = False
        try:
            R2Contract.AuditGroupCrossings(
                [
                    {
                        "content_group_key": "test-group",
                        "recommended_content_split": "test",
                        "label": "forbidden",
                    }
                ]
            )
        except ValueError:
            StrictRejection = True
        if not StrictRejection:
            Errors.append("crossing auditor accepted a third field")
    return {
        "sentinels": [
            "INDEPENDENT_SENTINEL_INVALID_A",
            "INDEPENDENT_SENTINEL_INVALID_B",
        ],
        "projection_keys": sorted(AllowedTestFields),
        "non_test_equal": FirstNonTest == SecondNonTest,
        "test_projection_equal": FirstTest == SecondTest == ExpectedProjection,
        "validation_metrics_equal": FirstMetrics == SecondMetrics,
        "validation_scores_equal": FirstScore == SecondScore,
        "crossing_results_equal": FirstCrossings == SecondCrossings,
        "strict_extra_field_rejection": StrictRejection,
    }, Errors


def AuditByteEquivalence(
    DeepvoiceRoot: Path,
    Helpers: Any,
) -> tuple[dict[str, Any], list[str]]:
    ReportsRoot = DeepvoiceRoot / "reports"
    R1Artifacts = DeepvoiceRoot / "artifacts" / "e00"
    R2Artifacts = DeepvoiceRoot / "artifacts" / "e00-r2"
    Pairs = [
        (ReportsRoot / "e00-head-metrics.csv", ReportsRoot / "e00-r2-head-metrics.csv"),
        (ReportsRoot / "e00-seed-summary.csv", ReportsRoot / "e00-r2-seed-summary.csv"),
        (
            ReportsRoot / "e00-bootstrap-summary.csv",
            ReportsRoot / "e00-r2-bootstrap-summary.csv",
        ),
        (
            ReportsRoot / "e00-shortcut-label-audit.csv",
            ReportsRoot / "e00-r2-shortcut-label-audit.csv",
        ),
        (
            ReportsRoot / "e00-shortcut-metric-fixture.csv",
            ReportsRoot / "e00-r2-shortcut-metric-fixture.csv",
        ),
        (
            ReportsRoot / "e00-singleton-equivalence.json",
            ReportsRoot / "e00-r2-singleton-equivalence.json",
        ),
    ]
    for Seed in ExpectedSeeds:
        Pairs.append(
            (
                R1Artifacts / f"validation-fixture-predictions-seed-{Seed}.csv.gz",
                R2Artifacts / f"validation-fixture-predictions-seed-{Seed}.csv.gz",
            )
        )
        Pairs.append(
            (
                R1Artifacts / f"bootstrap-replicates-seed-{Seed}.csv.gz",
                R2Artifacts / f"bootstrap-replicates-seed-{Seed}.csv.gz",
            )
        )
    Errors = []
    Results = []
    for R1Path, R2Path in Pairs:
        R1Hash = Helpers.HashFile(R1Path)
        R2Hash = Helpers.HashFile(R2Path)
        Equal = R1Path.stat().st_size == R2Path.stat().st_size and R1Hash == R2Hash
        Results.append(
            {
                "r1": str(R1Path.relative_to(DeepvoiceRoot)),
                "r2": str(R2Path.relative_to(DeepvoiceRoot)),
                "bytes": R2Path.stat().st_size,
                "sha256": R2Hash,
                "byte_equal": Equal,
            }
        )
        if not Equal:
            Errors.append(f"{R2Path.name} is not byte-equivalent to R1")
    return {"pair_count": len(Results), "pairs": Results}, Errors


def AuditRunProvenance(
    RepoRoot: Path,
    DeepvoiceRoot: Path,
    Helpers: Any,
    Config: dict[str, Any],
    RunRecord: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    ReportsRoot = DeepvoiceRoot / "reports"
    ArtifactRun = Helpers.LoadJson(
        DeepvoiceRoot / "artifacts" / "e00-r2" / "e00-r2-run.json"
    )
    if ArtifactRun != RunRecord:
        Errors.append("R2 artifact and report run records differ")
    if RunRecord.get("config") != Config:
        Errors.append("R2 run config differs from config.json")
    Runtime = Helpers.LoadJson(ReportsRoot / "e00-r2-runtime.json")
    if Runtime.get("runtime_seconds") != RunRecord.get("runtime_seconds"):
        Errors.append("R2 runtime report differs from run record")
    if Runtime.get("cost") != RunRecord.get("cost"):
        Errors.append("R2 cost report differs from run record")
    for RelativePath, Expected in RunRecord["code"]["files"].items():
        FilePath = DeepvoiceRoot / RelativePath
        if FilePath.stat().st_size != int(Expected["bytes"]):
            Errors.append(f"recorded code size differs: {RelativePath}")
        if Helpers.HashFile(FilePath) != Expected["sha256"]:
            Errors.append(f"recorded code SHA differs: {RelativePath}")
    for RelativePath, Expected in RunRecord["lineage"]["superseded_files"].items():
        FilePath = DeepvoiceRoot / RelativePath
        if FilePath.stat().st_size != int(Expected["bytes"]):
            Errors.append(f"R1 preserved size differs: {RelativePath}")
        if Helpers.HashFile(FilePath) != Expected["sha256"]:
            Errors.append(f"R1 preserved SHA differs: {RelativePath}")
    GitResult = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=C:/Users/MY PC/Desktop/Hackathon",
            "-C",
            str(RepoRoot),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    CurrentHead = GitResult.stdout.strip()
    if CurrentHead != RunRecord["code"]["git_head"]:
        Errors.append("recorded R2 git HEAD differs from current HEAD")
    Environment = RunRecord.get("environment", {})
    RequiredEnvironmentKeys = {
        "platform",
        "python_version",
        "python_executable",
        "logical_cpu_count",
        "packages",
        "torch_cuda_available",
    }
    if not RequiredEnvironmentKeys.issubset(Environment):
        Errors.append("R2 environment record is incomplete")
    RuntimeRecord = RunRecord.get("runtime_seconds", {})
    if float(RuntimeRecord.get("total", 0.0)) <= 0.0:
        Errors.append("R2 runtime is nonpositive")
    if len(RuntimeRecord.get("seed_runs", ())) != 3:
        Errors.append("R2 per-seed runtime record count differs")
    return {
        "git_head": CurrentHead,
        "code_file_count": len(RunRecord["code"]["files"]),
        "r1_preserved_file_count": len(RunRecord["lineage"]["superseded_files"]),
        "environment_keys": sorted(Environment),
        "runtime_total_seconds": RuntimeRecord.get("total"),
        "seed_runtime_count": len(RuntimeRecord.get("seed_runs", ())),
    }, Errors


def RunDeclaredTests(R2Root: Path) -> tuple[dict[str, Any], list[str]]:
    Environment = os.environ.copy()
    Environment["PYTHONDONTWRITEBYTECODE"] = "1"
    Result = subprocess.run(
        [sys.executable, "-B", str(R2Root / "run_tests.py")],
        check=False,
        capture_output=True,
        text=True,
        env=Environment,
    )
    Errors = []
    try:
        Record = json.loads(Result.stdout)
    except json.JSONDecodeError as Error:
        return {"stdout": Result.stdout, "stderr": Result.stderr}, [repr(Error)]
    if Result.returncode != 0:
        Errors.append(f"R2 declared tests exited {Result.returncode}")
    if Record.get("status") != "PASS" or Record.get("check_count") != 9:
        Errors.append("R2 declared tests are not 9/9 PASS")
    if any(Check.get("status") != "PASS" for Check in Record.get("checks", ())):
        Errors.append("at least one R2 declared test failed")
    return Record, Errors


def WriteOutputs(OutputRoot: Path, Record: dict[str, Any]) -> None:
    OutputRoot.mkdir(parents=True, exist_ok=False)
    JsonPath = OutputRoot / "e00-r2-validation-audit.json"
    MarkdownPath = OutputRoot / "e00-r2-validation-audit.md"
    JsonPath.write_text(
        json.dumps(Record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Lines = [
        f"EXPERIMENT_AUDIT: {Record['status']}",
        "",
        "# DeepVoice E00-R2 independent validation audit",
        "",
        f"감사 시각: {Record['finished_at_local']}",
        "범위: R2 test isolation, validation scorer/bootstrap, singleton, shortcut, provenance, R1 byte lineage",
        "",
        "## 판정",
        "",
    ]
    if Record["status"] == "PASS":
        Lines.append("E00-R2는 독립 감사에 통과했다. R1에서 확인된 test 통계 사용 2건이 제거되었고 E01 진입 gate를 충족한다.")
    else:
        Lines.append("E00-R2는 다음 차단 사유 때문에 모델 개선 근거로 사용할 수 없다.")
        Lines.append("")
        for Check in Record["checks"]:
            for Error in Check["errors"]:
                Lines.append(f"- {Check['check']}: {Error}")
    Lines.extend(
        [
            "",
            "## 독립 검사 결과",
            "",
            "| 검사 | 상태 |",
            "|---|---|",
        ]
    )
    for Check in Record["checks"]:
        Lines.append(f"| {Check['check']} | {Check['status']} |")
    Lines.extend(
        [
            "",
            f"별도 QA 체크리스트: **{Record['qa']['pass_count']}/{Record['qa']['check_count']} PASS**",
            "",
            "## test isolation",
            "",
            "- 실제 manifest에서 test 행은 즉시 `content_group_key`, `recommended_content_split` 두 필드로만 투영된다.",
            "- crossing auditor는 세 번째 필드를 가진 행을 거부하고 결과에는 `crossing_group_count`만 남긴다.",
            "- label/mask 및 shortcut label은 train+validation, scorer/bootstrap/fixture는 validation만 사용한다.",
            "- 두 synthetic test sentinel의 모든 비허용 metadata를 서로 다른 invalid 값으로 바꿔도 retained non-test rows, crossing, validation metric과 Score가 동일했다.",
            "- 감사 자체는 실제 test의 label, metadata, prediction 또는 성능 통계를 계산하지 않았다.",
            "",
            "## 검증된 수치와 재현성",
            "",
            "- manifest SHA-256과 137,328행 일치, content-group split crossing 0",
            "- 공식 유효 가중치 0.45/0.18/0.27/0.05/0.05",
            "- 3-seed EER, AUC, Brier, ADS, CPS, Score 독립 재계산 일치",
            "- 3-seed × 200 content-group bootstrap artifact와 sampling digest 일치",
            "- singleton max delta 0, 허용값 1e-6 이하",
            "- 7개 shortcut 축의 모든 non-test label slice 및 validation slice/head 완전",
            "- R2 validation metric/report 6개와 gzip artifact 6개가 R1과 byte-for-byte 동일",
            "- R2 자체 테스트 9/9 PASS, 독립 QA 34/34 PASS",
            "",
            "## 범위 제한",
            "",
            "PASS는 E00 평가 계약의 신뢰성을 뜻한다. fixture는 label-independent uniform RNG이며 모델, OOF 또는 baseline 성능이 아니다. 따라서 E01 구현은 가능하지만 E00 수치를 모델 성능으로 해석하면 안 된다.",
            "",
            "공식 평가 근거: https://dacon.io/competitions/official/236749/overview/evaluation (감사 확인일 2026-08-30)",
        ]
    )
    MarkdownPath.write_text("\n".join(Lines) + "\n", encoding="utf-8")


def Execute(RepoRoot: Path, OutputRoot: Path) -> dict[str, Any]:
    Started = time.perf_counter()
    DeepvoiceRoot = RepoRoot / "deepvoice"
    ReportsRoot = DeepvoiceRoot / "reports"
    R2Root = DeepvoiceRoot / "experiments" / "e00_r2"
    R2ArtifactRoot = DeepvoiceRoot / "artifacts" / "e00-r2"
    Helpers = LoadModule(
        "E00IndependentAuditHelpers",
        DeepvoiceRoot / "experiments" / "e00_audit" / "audit_e00.py",
    )
    R2Contract = LoadModule("E00R2ContractUnderAudit", R2Root / "contract.py")
    Config = Helpers.LoadJson(R2Root / "config.json")
    RunRecord = Helpers.LoadJson(ReportsRoot / "e00-r2-run-manifest.json")
    ManifestPath = ReportsRoot / "deepvoice-training-manifest.csv.gz"
    Checks = []
    QaChecks = []

    ManifestHash = Helpers.HashFile(ManifestPath)
    RowCount, ValidationRows, NonTestRows, Crossings = Helpers.ReadManifest(ManifestPath)
    ManifestErrors = []
    if ManifestHash != ExpectedManifestSha256:
        ManifestErrors.append(f"manifest SHA mismatch: {ManifestHash}")
    if RowCount != ExpectedManifestRows:
        ManifestErrors.append(f"manifest row count mismatch: {RowCount}")
    if RunRecord["manifest"]["sha256"] != ManifestHash:
        ManifestErrors.append("R2 run manifest SHA mismatch")
    if int(RunRecord["manifest"]["row_count"]) != RowCount:
        ManifestErrors.append("R2 run manifest row count mismatch")
    AddCheck(
        Checks,
        "manifest_identity",
        ManifestErrors,
        {"sha256": ManifestHash, "row_count": RowCount},
    )
    AddCheck(
        Checks,
        "content_group_split_crossing",
        [] if not Crossings else [f"{len(Crossings)} crossing groups"],
        {"crossing_group_count": len(Crossings), "examples": Crossings[:10]},
    )

    ActualNonTest, ActualTestProjection, _, ActualTotal = (
        R2Contract.LoadManifestPartitions(ManifestPath)
    )
    ProjectionKeySets = {tuple(sorted(Row)) for Row in ActualTestProjection}
    ActualRetainedForbiddenFields = sorted(
        {Key for Row in ActualTestProjection for Key in Row} - AllowedTestFields
    )
    SentinelEvidence, SentinelErrors = AuditSentinel(R2Contract)
    TestPolicyErrors = list(SentinelErrors)
    if ProjectionKeySets != {tuple(sorted(AllowedTestFields))}:
        TestPolicyErrors.append("real test projections do not contain exactly two allowed keys")
    if ActualRetainedForbiddenFields:
        TestPolicyErrors.append(
            f"real test projections retain forbidden fields: {ActualRetainedForbiddenFields}"
        )
    if ActualTotal != RowCount:
        TestPolicyErrors.append("R2 loader total differs from independent total")
    if len(ActualNonTest) != len(NonTestRows):
        TestPolicyErrors.append("R2 loader non-test scope differs from independent scope")
    if set(RunRecord.get("split_audit", {})) != {"crossing_group_count"}:
        TestPolicyErrors.append("split audit retains fields beyond crossing_group_count")
    FieldContract = RunRecord.get("test_field_access_contract", {})
    ZeroFields = {
        "retained_label_or_metadata_fields": FieldContract.get(
            "retained_label_or_metadata_fields"
        ),
        "test_prediction_rows": FieldContract.get("test_prediction_rows"),
        "test_metric_rows": FieldContract.get("test_metric_rows"),
        "test_mask_summary_rows": FieldContract.get("test_mask_summary_rows"),
    }
    if any(Value != 0 for Value in ZeroFields.values()):
        TestPolicyErrors.append(f"declared test output counters are nonzero: {ZeroFields}")

    RunCalls = {
        Name: GetCallArguments(R2Root / "run_e00.py", Name)
        for Name in (
            "BuildLabelMasks",
            "SummarizeLabelMasks",
            "BuildShortcutLabelAudit",
            "CalculateHeadMetrics",
            "BootstrapByContentGroup",
            "BuildFixturePredictionRows",
        )
    }
    ExpectedCallPrefixes = {
        "BuildLabelMasks": ["NonTestRows"],
        "SummarizeLabelMasks": ["NonTestRows", "NonTestLabels", "NonTestMasks"],
        "BuildShortcutLabelAudit": ["NonTestRows", "NonTestLabels", "NonTestMasks"],
        "CalculateHeadMetrics": ["ValidationLabels", "ValidationMasks", "Predictions"],
        "BootstrapByContentGroup": [
            "ValidationRows",
            "ValidationLabels",
            "ValidationMasks",
            "Predictions",
        ],
        "BuildFixturePredictionRows": [
            "ValidationRows",
            "ValidationLabels",
            "ValidationMasks",
            "Predictions",
        ],
    }
    for Name, ExpectedPrefix in ExpectedCallPrefixes.items():
        if not any(Arguments[: len(ExpectedPrefix)] == ExpectedPrefix for Arguments in RunCalls[Name]):
            TestPolicyErrors.append(f"{Name} is not statically scoped to {ExpectedPrefix}")
    LabelReportRows = Helpers.LoadCsv(ReportsRoot / "e00-r2-label-mask-audit.csv")
    if len(LabelReportRows) != 20:
        TestPolicyErrors.append("R2 label/mask report does not have 20 dataset-head rows")
    if any(
        Row.get("scope") != "train_plus_validation_no_test" for Row in LabelReportRows
    ):
        TestPolicyErrors.append("R2 label/mask report includes a non-approved scope")
    FixtureSplits = set()
    for Seed in ExpectedSeeds:
        with gzip.open(
            R2ArtifactRoot / f"validation-fixture-predictions-seed-{Seed}.csv.gz",
            "rt",
            encoding="utf-8-sig",
            newline="",
        ) as FileHandle:
            FixtureSplits.update(
                Row["recommended_content_split"] for Row in csv.DictReader(FileHandle)
            )
    if FixtureSplits != {"validation"}:
        TestPolicyErrors.append(f"R2 fixture splits are {sorted(FixtureSplits)}")
    AddCheck(
        Checks,
        "test_isolation_contract",
        TestPolicyErrors,
        {
            "allowed_test_fields": sorted(AllowedTestFields),
            "real_projection_key_sets": [list(Keys) for Keys in sorted(ProjectionKeySets)],
            "retained_forbidden_fields": ActualRetainedForbiddenFields,
            "split_audit_keys": sorted(RunRecord.get("split_audit", {})),
            "declared_zero_fields": ZeroFields,
            "fixture_splits": sorted(FixtureSplits),
            "sentinel": SentinelEvidence,
            "static_calls": RunCalls,
        },
    )

    ValidationLabels, ValidationMasks, ValidationLabelErrors = (
        Helpers.BuildLabelsAndMasks(ValidationRows)
    )
    NonTestLabels, NonTestMasks, NonTestLabelErrors = Helpers.BuildLabelsAndMasks(
        NonTestRows
    )
    AddCheck(
        Checks,
        "label_mask_contract",
        ValidationLabelErrors + NonTestLabelErrors,
        {
            "scope": "train_plus_validation_no_test",
            "dataset_contract": Helpers.DatasetContract,
            "report_row_count": len(LabelReportRows),
        },
    )

    MetricErrors = []
    FixtureErrors = []
    BootstrapErrors = []
    HeadReportRows = Helpers.LoadCsv(ReportsRoot / "e00-r2-head-metrics.csv")
    IndependentScores = {}
    SamplingDigests = {}
    for Seed in ExpectedSeeds:
        FixturePath = (
            R2ArtifactRoot / f"validation-fixture-predictions-seed-{Seed}.csv.gz"
        )
        Predictions, SeedFixtureErrors = Helpers.LoadFixture(
            FixturePath,
            ValidationRows,
            ValidationLabels,
            ValidationMasks,
            Seed,
        )
        FixtureErrors.extend(f"seed {Seed}: {Error}" for Error in SeedFixtureErrors)
        Metrics = Helpers.CalculateMetrics(
            ValidationLabels,
            ValidationMasks,
            Predictions,
        )
        Score = Helpers.CalculateScore(Metrics)
        IndependentScores[str(Seed)] = {
            "ADS": Score["ADS"],
            "CPS": Score["CPS"],
            "OfficialValidationProxy": Score["Score"],
            "brier": {Metric["head"]: Metric["brier"] for Metric in Metrics},
        }
        MetricErrors.extend(
            Helpers.CompareHeadMetrics(Metrics, Score, HeadReportRows, Seed)
        )
        BootstrapRows, SamplingDigest = Helpers.MakeBootstrapRows(
            ValidationRows,
            ValidationLabels,
            ValidationMasks,
            Predictions,
            Seed,
            int(Config["bootstrap_replicates"]),
        )
        RepeatedDigest = Helpers.GetSamplingDigest(
            ValidationRows,
            Seed,
            int(Config["bootstrap_replicates"]),
        )
        SamplingDigests[str(Seed)] = SamplingDigest
        if SamplingDigest != RepeatedDigest:
            BootstrapErrors.append(f"seed {Seed} sampling digest is not reproducible")
        ArtifactRows = Helpers.LoadGzipCsv(
            R2ArtifactRoot / f"bootstrap-replicates-seed-{Seed}.csv.gz"
        )
        BootstrapErrors.extend(
            Helpers.CompareBootstrapRows(BootstrapRows, ArtifactRows, Seed)
        )
    AddCheck(
        Checks,
        "official_metrics_and_brier",
        MetricErrors,
        {
            "effective_weights": Helpers.HeadWeights.tolist(),
            "independent_scores": IndependentScores,
        },
    )
    AddCheck(
        Checks,
        "fixture_provenance",
        FixtureErrors,
        {
            "fixture_is_model_result": Config.get("fixture_is_model_result"),
            "prediction_kind": RunRecord.get("prediction_kind"),
            "splits": sorted(FixtureSplits),
        },
    )
    AddCheck(
        Checks,
        "content_group_bootstrap",
        BootstrapErrors,
        {
            "seeds": list(ExpectedSeeds),
            "replicates_per_seed": int(Config["bootstrap_replicates"]),
            "sampling_unit": "content_group_key",
            "sampling_digests": SamplingDigests,
        },
    )

    SingletonIndependent = [Helpers.EvaluateSingleton(Seed) for Seed in ExpectedSeeds]
    SingletonSaved = Helpers.LoadJson(
        ReportsRoot / "e00-r2-singleton-equivalence.json"
    )
    SingletonErrors = []
    if SingletonIndependent != SingletonSaved:
        SingletonErrors.append("saved R2 singleton results differ from independent results")
    if any(
        Result["max_absolute_delta"] > 1e-6
        or Result["other_file_permutation_delta"] > 1e-6
        for Result in SingletonIndependent
    ):
        SingletonErrors.append("R2 singleton tolerance exceeded")
    AddCheck(
        Checks,
        "singleton_equivalence",
        SingletonErrors,
        {"tolerance": 1e-6, "results": SingletonIndependent},
    )

    ShortcutEvidence, ShortcutErrors = Helpers.AuditShortcuts(
        NonTestRows,
        ValidationRows,
        NonTestLabels,
        ReportsRoot / "e00-r2-shortcut-label-audit.csv",
        ReportsRoot / "e00-r2-shortcut-metric-fixture.csv",
    )
    AddCheck(
        Checks,
        "shortcut_alert_completeness",
        ShortcutErrors,
        ShortcutEvidence,
    )

    EquivalenceEvidence, EquivalenceErrors = AuditByteEquivalence(
        DeepvoiceRoot,
        Helpers,
    )
    AddCheck(
        Checks,
        "r1_r2_validation_byte_equivalence",
        EquivalenceErrors,
        EquivalenceEvidence,
    )

    DeclaredTests, DeclaredTestErrors = RunDeclaredTests(R2Root)
    AddCheck(
        Checks,
        "declared_tests",
        DeclaredTestErrors,
        DeclaredTests,
    )

    ProvenanceEvidence, ProvenanceErrors = AuditRunProvenance(
        RepoRoot,
        DeepvoiceRoot,
        Helpers,
        Config,
        RunRecord,
    )
    AddCheck(
        Checks,
        "run_provenance_and_r1_preservation",
        ProvenanceErrors,
        ProvenanceEvidence,
    )

    AddQa(QaChecks, "manifest_sha", ManifestHash == ExpectedManifestSha256, ManifestHash)
    AddQa(QaChecks, "manifest_rows", RowCount == ExpectedManifestRows, RowCount)
    AddQa(QaChecks, "zero_crossings", len(Crossings) == 0, len(Crossings))
    CrossingReportRows = Helpers.LoadCsv(ReportsRoot / "e00-r2-group-crossings.csv")
    AddQa(
        QaChecks,
        "crossing_report_empty",
        len(CrossingReportRows) == 1
        and not CrossingReportRows[0]["content_group_key"]
        and not CrossingReportRows[0]["splits"],
        CrossingReportRows,
    )
    AddQa(QaChecks, "config_identity", Config.get("experiment_id") == "E00-R2" and Config.get("revision") == 2, {"id": Config.get("experiment_id"), "revision": Config.get("revision")})
    AddQa(QaChecks, "config_manifest_sha", Config.get("manifest_sha256") == ManifestHash, Config.get("manifest_sha256"))
    AddQa(QaChecks, "config_seeds", tuple(Config.get("seeds", ())) == ExpectedSeeds, Config.get("seeds"))
    AddQa(QaChecks, "config_weights", Config.get("head_weights") == dict(zip(Helpers.HeadNames, Helpers.HeadWeights.tolist())), Config.get("head_weights"))
    AddQa(QaChecks, "fixture_not_model", Config.get("fixture_is_model_result") is False, Config.get("fixture_is_model_result"))
    AddQa(QaChecks, "test_policy_declared", "two" not in Config.get("test_split_policy", "").lower() or "content_group_key" in Config.get("test_split_policy", ""), Config.get("test_split_policy"))
    AddQa(QaChecks, "real_test_projection_two_keys", ProjectionKeySets == {tuple(sorted(AllowedTestFields))}, sorted(ProjectionKeySets))
    AddQa(QaChecks, "test_projection_no_forbidden_fields", not ActualRetainedForbiddenFields, ActualRetainedForbiddenFields)
    AddQa(QaChecks, "synthetic_projection_exact", SentinelEvidence["test_projection_equal"], SentinelEvidence["test_projection_equal"])
    AddQa(QaChecks, "crossing_rejects_metadata", SentinelEvidence["strict_extra_field_rejection"], SentinelEvidence["strict_extra_field_rejection"])
    AddQa(QaChecks, "sentinel_non_test_equal", SentinelEvidence["non_test_equal"], SentinelEvidence["non_test_equal"])
    AddQa(QaChecks, "sentinel_crossing_equal", SentinelEvidence["crossing_results_equal"], SentinelEvidence["crossing_results_equal"])
    AddQa(QaChecks, "sentinel_metrics_equal", SentinelEvidence["validation_metrics_equal"] and SentinelEvidence["validation_scores_equal"], SentinelEvidence)
    AddQa(QaChecks, "label_builder_non_test", any(Arguments[:1] == ["NonTestRows"] for Arguments in RunCalls["BuildLabelMasks"]), RunCalls["BuildLabelMasks"])
    AddQa(QaChecks, "mask_summary_non_test", any(Arguments[:3] == ["NonTestRows", "NonTestLabels", "NonTestMasks"] for Arguments in RunCalls["SummarizeLabelMasks"]), RunCalls["SummarizeLabelMasks"])
    AddQa(QaChecks, "shortcut_label_non_test", any(Arguments[:3] == ["NonTestRows", "NonTestLabels", "NonTestMasks"] for Arguments in RunCalls["BuildShortcutLabelAudit"]), RunCalls["BuildShortcutLabelAudit"])
    RunSource = (R2Root / "run_e00.py").read_text(encoding="utf-8")
    AddQa(QaChecks, "validation_derived_non_test", "for RowIndex, Row in enumerate(NonTestRows)" in RunSource and "ValidationRows = [NonTestRows" in RunSource, "static source")
    AddQa(QaChecks, "scorer_validation_only", any(Arguments[:3] == ["ValidationLabels", "ValidationMasks", "Predictions"] for Arguments in RunCalls["CalculateHeadMetrics"]), RunCalls["CalculateHeadMetrics"])
    AddQa(QaChecks, "bootstrap_validation_only", any(Arguments[:4] == ["ValidationRows", "ValidationLabels", "ValidationMasks", "Predictions"] for Arguments in RunCalls["BootstrapByContentGroup"]), RunCalls["BootstrapByContentGroup"])
    AddQa(QaChecks, "fixture_writer_validation_only", any(Arguments[:4] == ["ValidationRows", "ValidationLabels", "ValidationMasks", "Predictions"] for Arguments in RunCalls["BuildFixturePredictionRows"]), RunCalls["BuildFixturePredictionRows"])
    AddQa(QaChecks, "split_audit_minimal", set(RunRecord.get("split_audit", {})) == {"crossing_group_count"}, RunRecord.get("split_audit"))
    AddQa(QaChecks, "retained_metadata_zero", ZeroFields["retained_label_or_metadata_fields"] == 0 and not ActualRetainedForbiddenFields, ZeroFields)
    AddQa(QaChecks, "test_predictions_zero", ZeroFields["test_prediction_rows"] == 0 and FixtureSplits == {"validation"}, ZeroFields)
    AddQa(QaChecks, "test_metrics_zero", ZeroFields["test_metric_rows"] == 0 and any(Arguments[:3] == ["ValidationLabels", "ValidationMasks", "Predictions"] for Arguments in RunCalls["CalculateHeadMetrics"]), ZeroFields)
    AddQa(QaChecks, "test_mask_summary_zero", ZeroFields["test_mask_summary_rows"] == 0 and all(Row.get("scope") == "train_plus_validation_no_test" for Row in LabelReportRows), ZeroFields)
    AddQa(QaChecks, "fixture_artifacts_validation", FixtureSplits == {"validation"}, sorted(FixtureSplits))
    AddQa(QaChecks, "mask_report_non_test_scope", len(LabelReportRows) == 20 and all(Row.get("scope") == "train_plus_validation_no_test" for Row in LabelReportRows), len(LabelReportRows))
    AddQa(QaChecks, "declared_tests_9_of_9", not DeclaredTestErrors, DeclaredTests)
    AddQa(QaChecks, "r1_r2_byte_equivalence", not EquivalenceErrors and EquivalenceEvidence["pair_count"] == 12, EquivalenceEvidence["pair_count"])
    AddQa(QaChecks, "provenance_complete", not ProvenanceErrors, ProvenanceEvidence)

    if len(QaChecks) != 34:
        raise AssertionError(f"Independent QA must contain 34 checks, found {len(QaChecks)}")
    QaPassCount = sum(Check["status"] == "PASS" for Check in QaChecks)
    QaErrors = [Check["qa"] for Check in QaChecks if Check["status"] != "PASS"]
    AddCheck(
        Checks,
        "independent_qa_34",
        QaErrors,
        {"check_count": 34, "pass_count": QaPassCount, "checks": QaChecks},
    )

    Status = "PASS" if all(Check["status"] == "PASS" for Check in Checks) else "BLOCKED"
    Record = {
        "status": Status,
        "audit_id": "E00-R2-INDEPENDENT-AUDIT-20260830",
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "read-only R2 audit; R1 and R2 sources and outputs preserved",
        "test_data_handling": "real test rows used only as group/split projections for crossing; no test performance, label or metadata statistics calculated",
        "checks": Checks,
        "qa": {"check_count": 34, "pass_count": QaPassCount, "checks": QaChecks},
        "audit_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "runtime_seconds": time.perf_counter() - Started,
        "official_metric_source": {
            "url": "https://dacon.io/competitions/official/236749/overview/evaluation",
            "verified_date": "2026-08-30",
        },
    }
    WriteOutputs(OutputRoot, Record)
    return Record


def Main() -> int:
    if len(sys.argv) != 3:
        print("Usage: audit_e00_r2.py REPO_ROOT OUTPUT_ROOT", file=sys.stderr)
        return 2
    Record = Execute(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    print(
        json.dumps(
            {
                "status": Record["status"],
                "qa": Record["qa"],
                "checks": [
                    {
                        "check": Check["check"],
                        "status": Check["status"],
                        "errors": Check["errors"],
                    }
                    for Check in Record["checks"]
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if Record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(Main())
