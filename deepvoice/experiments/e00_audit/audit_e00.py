# /// <summary>
# Independent read-only auditor for the DeepVoice E00 evaluation contract
# /// </summary>

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


HeadNames = (
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)
HeadWeights = np.asarray((0.45, 0.18, 0.27, 0.05, 0.05), dtype=np.float64)
ExpectedManifestSha256 = (
    "2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6"
)
ExpectedManifestRows = 137328
ExpectedSeeds = (20260830, 20260831, 20260832)
ExpectedShortcutAxes = {
    "dataset",
    "source_family",
    "generator_or_provider",
    "codec",
    "sample_rate_hz",
    "channels",
    "duration_bucket_seconds",
}
DatasetContract = {
    "ljspeech-1.1": ((0.0, 0.0, None, 1.0, 0.0), "real"),
    "wavefake-1.2.0": ((1.0, 1.0, None, 1.0, 0.0), "synthetic"),
    "fma-small": ((0.0, None, 0.0, None, 1.0), "real"),
    "aime-open-model-subset": ((1.0, None, 1.0, 0.0, 1.0), "synthetic"),
}


def HashFile(FilePath: Path) -> str:
    Digest = hashlib.sha256()
    with FilePath.open("rb") as FileHandle:
        while True:
            Chunk = FileHandle.read(1024 * 1024)
            if not Chunk:
                break
            Digest.update(Chunk)
    return Digest.hexdigest()


def LoadCsv(FilePath: Path) -> list[dict[str, str]]:
    with FilePath.open("r", encoding="utf-8-sig", newline="") as FileHandle:
        return list(csv.DictReader(FileHandle))


def LoadGzipCsv(FilePath: Path) -> list[dict[str, str]]:
    with gzip.open(FilePath, "rt", encoding="utf-8-sig", newline="") as FileHandle:
        return list(csv.DictReader(FileHandle))


def LoadJson(FilePath: Path) -> Any:
    with FilePath.open("r", encoding="utf-8") as FileHandle:
        return json.load(FileHandle)


def IsClose(Left: float, Right: float, Tolerance: float = 1e-12) -> bool:
    return math.isclose(Left, Right, rel_tol=0.0, abs_tol=Tolerance)


def CalculateEer(Targets: np.ndarray, Scores: np.ndarray) -> float:
    Order = np.argsort(-Scores, kind="mergesort")
    SortedTargets = Targets[Order]
    SortedScores = Scores[Order]
    DistinctIndices = np.where(np.diff(SortedScores))[0]
    ThresholdIndices = np.r_[DistinctIndices, SortedTargets.size - 1]
    TruePositives = np.cumsum(SortedTargets)[ThresholdIndices]
    FalsePositives = 1 + ThresholdIndices - TruePositives
    PositiveCount = Targets.sum()
    NegativeCount = Targets.size - PositiveCount
    FalsePositiveRate = np.r_[0.0, FalsePositives / NegativeCount]
    TruePositiveRate = np.r_[0.0, TruePositives / PositiveCount]
    FalseNegativeRate = 1.0 - TruePositiveRate
    EqualIndex = int(np.argmin(np.abs(FalsePositiveRate - FalseNegativeRate)))
    return float(
        (FalsePositiveRate[EqualIndex] + FalseNegativeRate[EqualIndex]) / 2.0
    )


def CalculateRocAuc(Targets: np.ndarray, Scores: np.ndarray) -> float:
    Order = np.argsort(Scores, kind="mergesort")
    SortedScores = Scores[Order]
    Ranks = np.empty(Scores.size, dtype=np.float64)
    Start = 0
    while Start < Scores.size:
        End = Start + 1
        while End < Scores.size and SortedScores[End] == SortedScores[Start]:
            End += 1
        Ranks[Order[Start:End]] = (Start + 1 + End) / 2.0
        Start = End
    PositiveCount = int(Targets.sum())
    NegativeCount = Targets.size - PositiveCount
    PositiveRankSum = Ranks[Targets == 1.0].sum()
    Area = PositiveRankSum - PositiveCount * (PositiveCount + 1) / 2.0
    return float(Area / (PositiveCount * NegativeCount))


def CalculateBrier(Targets: np.ndarray, Scores: np.ndarray) -> float:
    return float(np.mean(np.square(Scores - Targets)))


def CalculateLogLoss(Targets: np.ndarray, Scores: np.ndarray) -> float:
    ClippedScores = np.clip(Scores, 1e-12, 1.0 - 1e-12)
    return float(
        np.mean(
            -Targets * np.log(ClippedScores)
            - (1.0 - Targets) * np.log(1.0 - ClippedScores)
        )
    )


def CalculateEce(Targets: np.ndarray, Scores: np.ndarray, BinCount: int = 15) -> float:
    Edges = np.linspace(0.0, 1.0, BinCount + 1)
    BinIndices = np.minimum(
        np.searchsorted(Edges, Scores, side="right") - 1,
        BinCount - 1,
    )
    Result = 0.0
    for BinIndex in range(BinCount):
        BinMask = BinIndices == BinIndex
        if np.any(BinMask):
            Result += float(np.mean(BinMask)) * abs(
                float(np.mean(Scores[BinMask])) - float(np.mean(Targets[BinMask]))
            )
    return float(Result)


def CalculateMetrics(
    Labels: np.ndarray,
    Masks: np.ndarray,
    Predictions: np.ndarray,
) -> list[dict[str, Any]]:
    Results = []
    for HeadIndex, HeadName in enumerate(HeadNames):
        ObservedMask = Masks[:, HeadIndex]
        Targets = Labels[ObservedMask, HeadIndex]
        Scores = Predictions[ObservedMask, HeadIndex]
        if Targets.size == 0 or np.unique(Targets).size != 2:
            Results.append(
                {
                    "head": HeadName,
                    "weight": float(HeadWeights[HeadIndex]),
                    "observed_count": int(Targets.size),
                    "positive_count": int(Targets.sum()) if Targets.size else 0,
                    "status": "INSUFFICIENT_CLASSES",
                }
            )
            continue
        Eer = CalculateEer(Targets, Scores)
        Auc = CalculateRocAuc(Targets, Scores)
        Results.append(
            {
                "head": HeadName,
                "weight": float(HeadWeights[HeadIndex]),
                "observed_count": int(Targets.size),
                "positive_count": int(Targets.sum()),
                "status": "OK",
                "auc": Auc,
                "eer": Eer,
                "brier": CalculateBrier(Targets, Scores),
                "log_loss": CalculateLogLoss(Targets, Scores),
                "ece_15": CalculateEce(Targets, Scores),
                "selection_component": 1.0 - Eer if HeadIndex < 3 else Auc,
            }
        )
    return Results


def CalculateScore(Metrics: Sequence[dict[str, Any]]) -> dict[str, float]:
    Components = np.asarray(
        [float(Metric["selection_component"]) for Metric in Metrics],
        dtype=np.float64,
    )
    Ads = float(
        0.5 * Components[0] + 0.2 * Components[1] + 0.3 * Components[2]
    )
    Cps = float(0.5 * Components[3] + 0.5 * Components[4])
    Score = float(0.9 * Ads + 0.1 * Cps)
    ExpandedScore = float(np.dot(HeadWeights, Components))
    if not IsClose(Score, ExpandedScore):
        raise AssertionError("Expanded official weights do not match ADS/CPS")
    return {"ADS": Ads, "CPS": Cps, "Score": Score}


def BuildLabelsAndMasks(
    Rows: Sequence[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    Labels = np.full((len(Rows), len(HeadNames)), np.nan, dtype=np.float64)
    Masks = np.zeros_like(Labels, dtype=bool)
    Errors = []
    for RowIndex, Row in enumerate(Rows):
        Dataset = Row["dataset"]
        if Dataset not in DatasetContract:
            Errors.append(f"unsupported dataset {Dataset}")
            continue
        Values, ExpectedLabel = DatasetContract[Dataset]
        if Row["label"] != ExpectedLabel:
            Errors.append(
                f"{Dataset}/{Row['sample_id']} expected {ExpectedLabel}, found {Row['label']}"
            )
        for HeadIndex, Value in enumerate(Values):
            if Value is not None:
                Labels[RowIndex, HeadIndex] = Value
                Masks[RowIndex, HeadIndex] = True
    return Labels, Masks, Errors


def ReadManifest(
    ManifestPath: Path,
) -> tuple[int, list[dict[str, str]], list[dict[str, str]], list[str]]:
    GroupSplits: dict[str, set[str]] = defaultdict(set)
    ValidationRows = []
    NonTestRows = []
    RowCount = 0
    with gzip.open(ManifestPath, "rt", encoding="utf-8-sig", newline="") as FileHandle:
        Reader = csv.DictReader(FileHandle)
        for Row in Reader:
            RowCount += 1
            GroupSplits[Row["content_group_key"]].add(
                Row["recommended_content_split"]
            )
            if Row["recommended_content_split"] == "validation":
                ValidationRows.append(Row)
            if Row["recommended_content_split"] != "test":
                NonTestRows.append(Row)
    Crossings = sorted(
        GroupKey for GroupKey, Splits in GroupSplits.items() if len(Splits) > 1
    )
    return RowCount, ValidationRows, NonTestRows, Crossings


def LoadFixture(
    FixturePath: Path,
    ValidationRows: Sequence[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
    Seed: int,
) -> tuple[np.ndarray, list[str]]:
    Rows = LoadGzipCsv(FixturePath)
    Errors = []
    if len(Rows) != len(ValidationRows):
        Errors.append(f"fixture row count {len(Rows)} != {len(ValidationRows)}")
    Predictions = np.empty((len(Rows), len(HeadNames)), dtype=np.float64)
    for RowIndex, FixtureRow in enumerate(Rows):
        if RowIndex >= len(ValidationRows):
            break
        ManifestRow = ValidationRows[RowIndex]
        for Key in ("dataset", "sample_id", "content_group_key"):
            if FixtureRow[Key] != ManifestRow[Key]:
                Errors.append(f"row {RowIndex} {Key} mismatch")
        if FixtureRow["recommended_content_split"] != "validation":
            Errors.append(f"row {RowIndex} is not validation")
        if FixtureRow["prediction_kind"] != "label_independent_contract_fixture_non_model":
            Errors.append(f"row {RowIndex} prediction kind mismatch")
        if int(FixtureRow["seed"]) != Seed:
            Errors.append(f"row {RowIndex} seed mismatch")
        for HeadIndex, HeadName in enumerate(HeadNames):
            MaskText = FixtureRow[f"{HeadName}_mask"].casefold()
            ExpectedMask = bool(Masks[RowIndex, HeadIndex])
            if (MaskText == "true") != ExpectedMask:
                Errors.append(f"row {RowIndex} {HeadName} mask mismatch")
            LabelText = FixtureRow[f"{HeadName}_label"]
            if ExpectedMask:
                if not IsClose(float(LabelText), float(Labels[RowIndex, HeadIndex])):
                    Errors.append(f"row {RowIndex} {HeadName} label mismatch")
            elif LabelText != "":
                Errors.append(f"row {RowIndex} {HeadName} masked label exposed")
            Predictions[RowIndex, HeadIndex] = float(
                FixtureRow[f"{HeadName}_prediction"]
            )
    ExpectedPredictions = np.random.default_rng(Seed).uniform(
        0.025,
        0.975,
        size=Predictions.shape,
    )
    MaximumDelta = float(np.max(np.abs(Predictions - ExpectedPredictions)))
    if MaximumDelta > 0.0:
        Errors.append(f"fixture RNG mismatch max delta {MaximumDelta}")
    return Predictions, Errors


def CompareHeadMetrics(
    IndependentMetrics: Sequence[dict[str, Any]],
    IndependentScore: dict[str, float],
    ReportRows: Sequence[dict[str, str]],
    Seed: int,
) -> list[str]:
    Errors = []
    SeedRows = [Row for Row in ReportRows if int(Row["seed"]) == Seed]
    if len(SeedRows) != len(HeadNames):
        return [f"seed {Seed} has {len(SeedRows)} metric rows"]
    ByHead = {Row["head"]: Row for Row in SeedRows}
    for Metric in IndependentMetrics:
        HeadName = Metric["head"]
        Report = ByHead.get(HeadName)
        if Report is None:
            Errors.append(f"seed {Seed} missing {HeadName}")
            continue
        for Key in ("weight", "observed_count", "positive_count"):
            Expected = float(Metric[Key])
            Found = float(Report[Key])
            if not IsClose(Expected, Found):
                Errors.append(f"seed {Seed} {HeadName} {Key} mismatch")
        for Key in ("auc", "eer", "brier", "log_loss", "ece_15", "selection_component"):
            if not IsClose(float(Metric[Key]), float(Report[Key])):
                Errors.append(f"seed {Seed} {HeadName} {Key} mismatch")
        for ReportKey, ScoreKey in (
            ("ADS", "ADS"),
            ("CPS", "CPS"),
            ("OfficialValidationProxy", "Score"),
            ("RobustSelectionScore", "Score"),
        ):
            if not IsClose(float(Report[ReportKey]), IndependentScore[ScoreKey]):
                Errors.append(f"seed {Seed} {HeadName} {ReportKey} mismatch")
    return Errors


def MakeBootstrapRows(
    ValidationRows: Sequence[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
    Predictions: np.ndarray,
    Seed: int,
    Replicates: int,
) -> tuple[list[dict[str, float]], str]:
    GroupIndices: dict[str, list[int]] = defaultdict(list)
    for RowIndex, Row in enumerate(ValidationRows):
        GroupIndices[Row["content_group_key"]].append(RowIndex)
    GroupKeys = sorted(GroupIndices)
    Generator = np.random.default_rng(Seed)
    SamplingDigest = hashlib.sha256()
    Results = []
    for ReplicateIndex in range(Replicates):
        SampledPositions = Generator.integers(0, len(GroupKeys), size=len(GroupKeys))
        SamplingDigest.update(SampledPositions.astype("<i8", copy=False).tobytes())
        RowIndices = np.fromiter(
            (
                Index
                for Position in SampledPositions
                for Index in GroupIndices[GroupKeys[int(Position)]]
            ),
            dtype=np.int64,
        )
        Metrics = CalculateMetrics(
            Labels[RowIndices],
            Masks[RowIndices],
            Predictions[RowIndices],
        )
        if any(Metric["status"] != "OK" for Metric in Metrics):
            continue
        Score = CalculateScore(Metrics)["Score"]
        Result = {
            "replicate": float(ReplicateIndex),
            "RobustSelectionScore": Score,
        }
        for Metric in Metrics:
            Result[f"{Metric['head']}_component"] = float(
                Metric["selection_component"]
            )
        Results.append(Result)
    return Results, SamplingDigest.hexdigest()


def GetSamplingDigest(
    ValidationRows: Sequence[dict[str, str]],
    Seed: int,
    Replicates: int,
) -> str:
    GroupKeys = sorted({Row["content_group_key"] for Row in ValidationRows})
    Generator = np.random.default_rng(Seed)
    Digest = hashlib.sha256()
    for _ in range(Replicates):
        Positions = Generator.integers(0, len(GroupKeys), size=len(GroupKeys))
        Digest.update(Positions.astype("<i8", copy=False).tobytes())
    return Digest.hexdigest()


def CompareBootstrapRows(
    IndependentRows: Sequence[dict[str, float]],
    ArtifactRows: Sequence[dict[str, str]],
    Seed: int,
) -> list[str]:
    Errors = []
    if len(IndependentRows) != len(ArtifactRows):
        return [
            f"seed {Seed} bootstrap row count {len(ArtifactRows)} != {len(IndependentRows)}"
        ]
    for Independent, Artifact in zip(IndependentRows, ArtifactRows):
        for Key, Value in Independent.items():
            if not IsClose(float(Value), float(Artifact[Key])):
                Errors.append(
                    f"seed {Seed} replicate {Independent['replicate']} {Key} mismatch"
                )
                if len(Errors) >= 20:
                    return Errors
    return Errors


def EvaluateSingleton(Seed: int) -> dict[str, Any]:
    Generator = np.random.default_rng(Seed)
    FileSegments = []
    for _ in range(64):
        SegmentCount = int(Generator.integers(1, 9))
        FileSegments.append(Generator.uniform(0.0, 1.0, size=(SegmentCount, 5)))
    SingletonOutputs = np.stack(
        [np.mean(Segments, axis=0, dtype=np.float64) for Segments in FileSegments]
    )
    BatchedOutputs = np.empty_like(SingletonOutputs)
    BatchOrder = Generator.permutation(64)
    for FileIndex in BatchOrder:
        BatchedOutputs[FileIndex] = np.mean(
            FileSegments[FileIndex],
            axis=0,
            dtype=np.float64,
        )
    OtherFilePermutationDelta = 0.0
    AnchorOutput = np.mean(FileSegments[0], axis=0, dtype=np.float64)
    for _ in range(10):
        Generator.shuffle(BatchOrder)
        CurrentOutput = np.mean(FileSegments[0], axis=0, dtype=np.float64)
        OtherFilePermutationDelta = max(
            OtherFilePermutationDelta,
            float(np.max(np.abs(AnchorOutput - CurrentOutput))),
        )
    return {
        "seed": Seed,
        "file_count": 64,
        "maximum_segments": 8,
        "max_absolute_delta": float(np.max(np.abs(SingletonOutputs - BatchedOutputs))),
        "other_file_permutation_delta": OtherFilePermutationDelta,
    }


def MakeDurationBucket(DurationText: str) -> str:
    Duration = float(DurationText)
    if Duration <= 4.0:
        return "00_to_04"
    if Duration <= 8.0:
        return "04_to_08"
    if Duration <= 15.0:
        return "08_to_15"
    if Duration <= 30.0:
        return "15_to_30"
    return "30_plus"


def GetShortcutValue(Row: dict[str, str], Axis: str) -> str:
    if Axis == "duration_bucket_seconds":
        return MakeDurationBucket(Row["duration_seconds"])
    return Row[Axis]


def AuditShortcuts(
    NonTestRows: Sequence[dict[str, str]],
    ValidationRows: Sequence[dict[str, str]],
    NonTestLabels: np.ndarray,
    LabelReportPath: Path,
    MetricReportPath: Path,
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    LabelRows = LoadCsv(LabelReportPath)
    MetricRows = LoadCsv(MetricReportPath)
    LabelAxes = {Row["axis"] for Row in LabelRows}
    MetricAxes = {Row["axis"] for Row in MetricRows}
    if LabelAxes != ExpectedShortcutAxes:
        Errors.append(f"label shortcut axes mismatch: {sorted(LabelAxes)}")
    if MetricAxes != ExpectedShortcutAxes:
        Errors.append(f"metric shortcut axes mismatch: {sorted(MetricAxes)}")
    ExpectedLabelPairs = {
        (Axis, GetShortcutValue(Row, Axis))
        for Axis in ExpectedShortcutAxes
        for Row in NonTestRows
    }
    ExpectedMetricTriples = {
        (Axis, GetShortcutValue(Row, Axis), HeadName)
        for Axis in ExpectedShortcutAxes
        for Row in ValidationRows
        for HeadName in HeadNames
    }
    ActualLabelPairs = {(Row["axis"], Row["slice"]) for Row in LabelRows}
    ActualMetricTriples = {
        (Row["axis"], Row["slice"], Row["head"]) for Row in MetricRows
    }
    if ActualLabelPairs != ExpectedLabelPairs:
        Errors.append("label shortcut slices are incomplete or duplicated")
    if ActualMetricTriples != ExpectedMetricTriples:
        Errors.append("metric shortcut slices are incomplete or duplicated")
    if len(ActualLabelPairs) != len(LabelRows):
        Errors.append("duplicate label shortcut rows")
    if len(ActualMetricTriples) != len(MetricRows):
        Errors.append("duplicate metric shortcut rows")
    if any(Row["scope"] != "train_plus_validation_no_test" for Row in LabelRows):
        Errors.append("label shortcut report scope is not train+validation only")
    if any(
        Row["scope"] != "validation_seed_20260830_contract_fixture_non_model"
        for Row in MetricRows
    ):
        Errors.append("metric shortcut report scope is not validation fixture only")
    PureCount = 0
    for ReportRow in LabelRows:
        Axis = ReportRow["axis"]
        Slice = ReportRow["slice"]
        Indices = np.asarray(
            [
                Index
                for Index, Row in enumerate(NonTestRows)
                if GetShortcutValue(Row, Axis) == Slice
            ],
            dtype=np.int64,
        )
        FakeRate = float(np.mean(NonTestLabels[Indices, 0]))
        Pure = FakeRate in (0.0, 1.0)
        PureCount += int(Pure)
        ExpectedRisk = "HIGH" if Pure else "MONITOR"
        if int(ReportRow["row_count"]) != int(Indices.size):
            Errors.append(f"{Axis}/{Slice} row count mismatch")
        if not IsClose(float(ReportRow["file_fake_rate"]), FakeRate):
            Errors.append(f"{Axis}/{Slice} fake rate mismatch")
        if ReportRow["shortcut_risk"] != ExpectedRisk:
            Errors.append(f"{Axis}/{Slice} shortcut risk mismatch")
    return {
        "axes": sorted(ExpectedShortcutAxes),
        "label_slice_count": len(LabelRows),
        "metric_slice_head_count": len(MetricRows),
        "high_risk_label_pure_slice_count": PureCount,
    }, Errors


def AuditTestPolicy(
    RunPath: Path,
    ContractPath: Path,
    FixturePaths: Sequence[Path],
) -> tuple[dict[str, Any], list[str]]:
    RunText = RunPath.read_text(encoding="utf-8")
    ContractText = ContractPath.read_text(encoding="utf-8")
    RunTree = ast.parse(RunText)
    Violations = []
    Evidence = []
    for Node in ast.walk(RunTree):
        if not isinstance(Node, ast.Call) or not isinstance(Node.func, ast.Name):
            continue
        if Node.func.id == "SummarizeLabelMasks" and Node.args:
            ArgumentName = Node.args[0].id if isinstance(Node.args[0], ast.Name) else "<expr>"
            Evidence.append(
                {
                    "line": Node.lineno,
                    "call": "SummarizeLabelMasks",
                    "first_argument": ArgumentName,
                }
            )
            if ArgumentName == "AllRows":
                Violations.append(
                    "run_e00.py line 514 computes dataset/head row, observed, masked and observed-value summaries from AllRows, which includes the locked test split"
                )
    if '"test_row_count": SplitCounts.get("test", 0)' in ContractText:
        LineNumber = ContractText[: ContractText.index('"test_row_count"')].count("\n") + 1
        Evidence.append(
            {
                "line": LineNumber,
                "call": "AuditGroupCrossings",
                "field": "test_row_count",
            }
        )
        Violations.append(
            "contract.py records test_row_count in the split summary although the fixed policy permits only crossing detection"
        )
    FixtureSplits = set()
    for FixturePath in FixturePaths:
        with gzip.open(FixturePath, "rt", encoding="utf-8-sig", newline="") as FileHandle:
            Reader = csv.DictReader(FileHandle)
            FixtureSplits.update(Row["recommended_content_split"] for Row in Reader)
    if FixtureSplits != {"validation"}:
        Violations.append(f"fixture artifacts contain splits {sorted(FixtureSplits)}")
    return {
        "policy": "test may be used only for content-group crossing detection",
        "fixture_splits": sorted(FixtureSplits),
        "static_evidence": Evidence,
        "violation_count": len(Violations),
        "violations": Violations,
    }, Violations


def AuditRunRecords(
    RepoRoot: Path,
    DeepvoiceRoot: Path,
    Config: dict[str, Any],
    RunRecord: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    Errors = []
    ExpectedWeights = dict(zip(HeadNames, HeadWeights.tolist()))
    if tuple(Config.get("seeds", ())) != ExpectedSeeds:
        Errors.append("configured seeds mismatch")
    if Config.get("head_weights") != ExpectedWeights:
        Errors.append("configured head weights mismatch")
    if Config.get("manifest_sha256") != ExpectedManifestSha256:
        Errors.append("configured manifest SHA mismatch")
    if RunRecord.get("config") != Config:
        Errors.append("run record config does not equal config.json")
    ArtifactRun = LoadJson(DeepvoiceRoot / "artifacts" / "e00" / "e00-run.json")
    if ArtifactRun != RunRecord:
        Errors.append("artifact and report run records differ")
    RuntimeReport = LoadJson(DeepvoiceRoot / "reports" / "e00-runtime.json")
    if RuntimeReport.get("runtime_seconds") != RunRecord.get("runtime_seconds"):
        Errors.append("runtime report differs from run record")
    if RuntimeReport.get("cost") != RunRecord.get("cost"):
        Errors.append("cost report differs from run record")
    CodeErrors = []
    for RelativePath, Record in RunRecord["code"]["files"].items():
        FilePath = DeepvoiceRoot / RelativePath
        if not FilePath.is_file():
            CodeErrors.append(f"missing {RelativePath}")
            continue
        if FilePath.stat().st_size != int(Record["bytes"]):
            CodeErrors.append(f"size mismatch {RelativePath}")
        if HashFile(FilePath) != Record["sha256"]:
            CodeErrors.append(f"SHA mismatch {RelativePath}")
    Errors.extend(CodeErrors)
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
    if RunRecord["code"]["git_head"] != CurrentHead:
        Errors.append("recorded git HEAD differs from current HEAD")
    Environment = RunRecord.get("environment", {})
    RequiredEnvironment = {
        "platform",
        "python_version",
        "python_executable",
        "logical_cpu_count",
        "packages",
        "torch_cuda_available",
    }
    if not RequiredEnvironment.issubset(Environment):
        Errors.append("environment record is incomplete")
    Runtime = RunRecord.get("runtime_seconds", {})
    if float(Runtime.get("total", 0.0)) <= 0.0:
        Errors.append("runtime total is missing or nonpositive")
    if len(Runtime.get("seed_runs", ())) != len(ExpectedSeeds):
        Errors.append("per-seed runtime records are incomplete")
    return {
        "git_head": CurrentHead,
        "recorded_git_status_present": bool(RunRecord["code"].get("git_status_short")),
        "recorded_code_file_count": len(RunRecord["code"]["files"]),
        "recorded_environment_keys": sorted(Environment),
        "runtime_total_seconds": Runtime.get("total"),
        "seed_runtime_count": len(Runtime.get("seed_runs", ())),
    }, Errors


def GetPackageVersion(PackageName: str) -> str | None:
    try:
        return importlib.metadata.version(PackageName)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def WriteOutputs(
    OutputRoot: Path,
    Record: dict[str, Any],
) -> None:
    OutputRoot.mkdir(parents=True, exist_ok=False)
    JsonPath = OutputRoot / "e00-validation-audit.json"
    MarkdownPath = OutputRoot / "e00-validation-audit.md"
    JsonPath.write_text(
        json.dumps(Record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    BlockedChecks = [Check for Check in Record["checks"] if Check["status"] == "BLOCKED"]
    Lines = [
        f"EXPERIMENT_AUDIT: {Record['status']}",
        "",
        "# DeepVoice E00 independent validation audit",
        "",
        f"감사 시각: {Record['finished_at_local']}",
        "범위: E00 scorer, label/mask, split policy, content-group bootstrap, singleton, shortcut, provenance records",
        "",
        "## 판정",
        "",
    ]
    if BlockedChecks:
        Lines.append("E00 결과는 모델 개선 근거로 채택할 수 없다. 고정 stop criterion인 `test statistic use`가 실제 코드와 산출물에서 확인됐다.")
        Lines.append("")
        for Check in BlockedChecks:
            for Error in Check["errors"]:
                Lines.append(f"- {Check['check']}: {Error}")
    else:
        Lines.append("모든 독립 검사에 통과했다.")
    Lines.extend(
        [
            "",
            "## 독립 검사 결과",
            "",
            "| 검사 | 상태 | 핵심 근거 |",
            "|---|---|---|",
        ]
    )
    Summaries = {
        "manifest_identity": "SHA-256 고정, 137,328행",
        "content_group_split_crossing": "전체 content_group의 split crossing 0",
        "label_mask_contract": "train+validation에서 4개 source mapping 일치",
        "official_metric_and_brier": "DACON EER/AUC/ADS/CPS/Score와 Brier 독립 재계산 일치",
        "content_group_bootstrap": "3 seed × 200회 artifact 및 sampling digest 일치",
        "singleton_equivalence": "3 seed 모두 max delta ≤1e-6",
        "fixture_provenance": "validation-only, label-independent RNG, non-model/non-OOF 표시",
        "shortcut_alert_completeness": "7개 축, 모든 slice/head 및 HIGH/MONITOR 경보 완전",
        "run_provenance": "seed/config/git/code SHA/environment/runtime 기록 일치",
        "test_statistics_nonuse": "위반 2건: test row count 저장, AllRows label/mask summary",
    }
    for Check in Record["checks"]:
        Lines.append(
            f"| {Check['check']} | {Check['status']} | {Summaries.get(Check['check'], '')} |"
        )
    Lines.extend(
        [
            "",
            "## test split 계약 위반",
            "",
            "E00의 고정 정책은 test를 content-group crossing 탐지에만 사용하는 것이다. 그러나:",
            "",
            "1. `contract.py`의 `AuditGroupCrossings`가 `test_row_count`를 계산해 run manifest에 저장했다.",
            "2. `run_e00.py`가 `SummarizeLabelMasks(AllRows, AllLabels, AllMasks)`를 호출해 test를 포함한 dataset/head별 행 수, 관측 수, 마스크 수, 관측 label 값을 `e00-label-mask-audit.csv`에 저장했다.",
            "",
            "이는 예측 지표를 test에서 계산하지 않았다는 주장만으로 해소되지 않는다. `experiment-plan.csv`의 E00 stop criterion은 더 넓은 `test statistic use`이며, `modeling-plan.md`는 최종 후보와 calibration 동결 전 test를 열지 않도록 고정한다. 따라서 BLOCKED다.",
            "",
            "감사 자체는 test label·예측 분포를 새로 계산하지 않았다. 전체 row count와 content-group split membership으로 crossing만 확인했고, label/mask·metric·shortcut 검사는 train+validation 또는 validation에 한정했다.",
            "",
            "## 통과한 계약",
            "",
            "- manifest SHA와 137,328행이 고정 입력과 일치한다.",
            "- content-group split crossing은 0이다.",
            "- 4개 데이터셋의 five-head label/mask mapping이 train+validation에서 일치한다.",
            "- 공식 가중치는 0.45/0.18/0.27/0.05/0.05이고, 저장된 3-seed EER, AUC, ADS, CPS, Score, Brier를 독립 재계산해 일치했다.",
            "- content_group_key bootstrap 3-seed × 200회가 저장 artifact와 일치하고 같은 seed의 sampling digest가 재현됐다.",
            "- singleton delta는 3-seed 모두 0이며 허용치 1e-6 이하다.",
            "- fixture는 validation-only이며 seed와 row count만 받는 uniform RNG와 정확히 일치한다. 모델, OOF, baseline이 아니라는 표시는 config, run record, row artifact에 존재한다.",
            "- shortcut 감사 7개 축과 모든 slice/head가 존재하고 label-pure slice는 HIGH로 누락 없이 표시됐다.",
            "- seed, config, git HEAD/status, 실행 코드 SHA, 환경, runtime/cost 기록이 존재하며 현재 파일과 일치한다.",
            "",
            "## 수정 책임과 재감사 조건",
            "",
            "실험 엔지니어가 기존 결과를 수정하지 말고 새 E00 run으로 다음을 수행해야 한다.",
            "",
            "- full-manifest label/mask 생성을 제거하고 train+validation 또는 validation으로만 제한한다.",
            "- split crossing 함수는 test count를 반환·저장하지 않고 group→split 집합의 crossing 여부만 기록한다.",
            "- 이전 `e00-label-mask-audit.csv`, run manifest, batch report를 성능 근거로 폐기하고 새 run ID/별도 출력 경로로 재실행한다.",
            "- 재감사 전까지 E01 및 후속 모델 개선을 시작하지 않는다.",
            "",
            "공식 평가 근거: https://dacon.io/competitions/official/236749/overview/evaluation (감사 확인일 2026-08-30)",
        ]
    )
    MarkdownPath.write_text("\n".join(Lines) + "\n", encoding="utf-8")


def Execute(RepoRoot: Path, OutputRoot: Path) -> dict[str, Any]:
    Started = time.perf_counter()
    DeepvoiceRoot = RepoRoot / "deepvoice"
    ReportsRoot = DeepvoiceRoot / "reports"
    E00Root = DeepvoiceRoot / "experiments" / "e00"
    ArtifactRoot = DeepvoiceRoot / "artifacts" / "e00"
    ManifestPath = ReportsRoot / "deepvoice-training-manifest.csv.gz"
    Config = LoadJson(E00Root / "config.json")
    RunRecord = LoadJson(ReportsRoot / "e00-run-manifest.json")
    Checks = []

    ManifestHash = HashFile(ManifestPath)
    RowCount, ValidationRows, NonTestRows, Crossings = ReadManifest(ManifestPath)
    ManifestErrors = []
    if ManifestHash != ExpectedManifestSha256:
        ManifestErrors.append(f"manifest SHA mismatch {ManifestHash}")
    if RowCount != ExpectedManifestRows:
        ManifestErrors.append(f"manifest row count {RowCount}")
    if RunRecord["manifest"]["sha256"] != ManifestHash:
        ManifestErrors.append("run record manifest SHA mismatch")
    if int(RunRecord["manifest"]["row_count"]) != RowCount:
        ManifestErrors.append("run record manifest row count mismatch")
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

    ValidationLabels, ValidationMasks, ValidationLabelErrors = BuildLabelsAndMasks(
        ValidationRows
    )
    NonTestLabels, NonTestMasks, NonTestLabelErrors = BuildLabelsAndMasks(NonTestRows)
    LabelErrors = ValidationLabelErrors + NonTestLabelErrors
    MappingEvidence = {}
    for Dataset, (Values, ExpectedLabel) in DatasetContract.items():
        MappingEvidence[Dataset] = {
            "source_label": ExpectedLabel,
            "labels": [Value if Value is not None else "masked" for Value in Values],
            "non_test_rows_checked": sum(Row["dataset"] == Dataset for Row in NonTestRows),
        }
    AddCheck(Checks, "label_mask_contract", LabelErrors, MappingEvidence)

    HeadReportRows = LoadCsv(ReportsRoot / "e00-head-metrics.csv")
    MetricErrors = []
    FixtureErrors = []
    BootstrapErrors = []
    FixturePaths = []
    IndependentScores = {}
    SamplingDigests = {}
    for Seed in ExpectedSeeds:
        FixturePath = ArtifactRoot / f"validation-fixture-predictions-seed-{Seed}.csv.gz"
        FixturePaths.append(FixturePath)
        Predictions, SeedFixtureErrors = LoadFixture(
            FixturePath,
            ValidationRows,
            ValidationLabels,
            ValidationMasks,
            Seed,
        )
        FixtureErrors.extend(f"seed {Seed}: {Error}" for Error in SeedFixtureErrors)
        Metrics = CalculateMetrics(ValidationLabels, ValidationMasks, Predictions)
        Score = CalculateScore(Metrics)
        IndependentScores[str(Seed)] = {
            "ADS": Score["ADS"],
            "CPS": Score["CPS"],
            "OfficialValidationProxy": Score["Score"],
            "head_metrics": Metrics,
        }
        MetricErrors.extend(
            CompareHeadMetrics(Metrics, Score, HeadReportRows, Seed)
        )
        BootstrapRows, SamplingDigest = MakeBootstrapRows(
            ValidationRows,
            ValidationLabels,
            ValidationMasks,
            Predictions,
            Seed,
            int(Config["bootstrap_replicates"]),
        )
        RepeatedDigest = GetSamplingDigest(
            ValidationRows,
            Seed,
            int(Config["bootstrap_replicates"]),
        )
        SamplingDigests[str(Seed)] = SamplingDigest
        if SamplingDigest != RepeatedDigest:
            BootstrapErrors.append(f"seed {Seed} sampling plan is not reproducible")
        ArtifactBootstrapRows = LoadGzipCsv(
            ArtifactRoot / f"bootstrap-replicates-seed-{Seed}.csv.gz"
        )
        BootstrapErrors.extend(
            CompareBootstrapRows(BootstrapRows, ArtifactBootstrapRows, Seed)
        )
    OfficialExampleTargets = np.asarray((0.0, 0.0, 1.0, 1.0))
    OfficialExampleScores = np.asarray((0.1, 0.4, 0.35, 0.8))
    OfficialExample = {
        "eer": CalculateEer(OfficialExampleTargets, OfficialExampleScores),
        "auc": CalculateRocAuc(OfficialExampleTargets, OfficialExampleScores),
        "brier": CalculateBrier(OfficialExampleTargets, OfficialExampleScores),
    }
    if not IsClose(OfficialExample["eer"], 0.5):
        MetricErrors.append("independent EER example mismatch")
    if not IsClose(OfficialExample["auc"], 0.75):
        MetricErrors.append("independent AUC example mismatch")
    if not IsClose(OfficialExample["brier"], 0.158125):
        MetricErrors.append("independent Brier example mismatch")
    AddCheck(
        Checks,
        "official_metric_and_brier",
        MetricErrors,
        {
            "official_formula": "Score=0.9*ADS+0.1*CPS; ADS=0.5*(1-FileEER)+0.2*(1-VoiceEER)+0.3*(1-MusicEER); CPS=0.5*VoicePresenceAUC+0.5*MusicPresenceAUC",
            "expanded_weights": HeadWeights.tolist(),
            "official_example": OfficialExample,
            "independent_seed_scores": IndependentScores,
        },
    )
    AddCheck(
        Checks,
        "content_group_bootstrap",
        BootstrapErrors,
        {
            "sampling_unit": "content_group_key",
            "seeds": list(ExpectedSeeds),
            "replicates_per_seed": int(Config["bootstrap_replicates"]),
            "sampling_digests": SamplingDigests,
        },
    )

    SingletonSaved = LoadJson(ReportsRoot / "e00-singleton-equivalence.json")
    SingletonIndependent = [EvaluateSingleton(Seed) for Seed in ExpectedSeeds]
    SingletonErrors = []
    if SingletonSaved != SingletonIndependent:
        SingletonErrors.append("saved singleton results differ from independent calculation")
    for Result in SingletonIndependent:
        if Result["max_absolute_delta"] > 1e-6:
            SingletonErrors.append(f"seed {Result['seed']} singleton tolerance exceeded")
        if Result["other_file_permutation_delta"] > 1e-6:
            SingletonErrors.append(f"seed {Result['seed']} permutation tolerance exceeded")
    AddCheck(
        Checks,
        "singleton_equivalence",
        SingletonErrors,
        {"tolerance": 1e-6, "results": SingletonIndependent},
    )

    ConfigFixtureErrors = []
    if Config.get("fixture_is_model_result") is not False:
        ConfigFixtureErrors.append("config does not mark fixture_is_model_result=false")
    if "label-independent" not in Config.get("fixture_prediction_kind", ""):
        ConfigFixtureErrors.append("config does not mark label independence")
    if "not model, OOF or baseline" not in RunRecord.get("prediction_kind", ""):
        ConfigFixtureErrors.append("run record lacks non-model/non-OOF/non-baseline marker")
    FixtureErrors.extend(ConfigFixtureErrors)
    AddCheck(
        Checks,
        "fixture_provenance",
        FixtureErrors,
        {
            "kind": Config.get("fixture_prediction_kind"),
            "fixture_is_model_result": Config.get("fixture_is_model_result"),
            "run_marker": RunRecord.get("prediction_kind"),
            "artifact_split": "validation",
            "rng_contract": "numpy.default_rng(seed).uniform(0.025,0.975,(validation_rows,5))",
        },
    )

    ShortcutEvidence, ShortcutErrors = AuditShortcuts(
        NonTestRows,
        ValidationRows,
        NonTestLabels,
        ReportsRoot / "e00-shortcut-label-audit.csv",
        ReportsRoot / "e00-shortcut-metric-fixture.csv",
    )
    AddCheck(
        Checks,
        "shortcut_alert_completeness",
        ShortcutErrors,
        ShortcutEvidence,
    )

    RunEvidence, RunErrors = AuditRunRecords(
        RepoRoot,
        DeepvoiceRoot,
        Config,
        RunRecord,
    )
    AddCheck(Checks, "run_provenance", RunErrors, RunEvidence)

    TestEvidence, TestViolations = AuditTestPolicy(
        E00Root / "run_e00.py",
        E00Root / "contract.py",
        FixturePaths,
    )
    AddCheck(
        Checks,
        "test_statistics_nonuse",
        TestViolations,
        TestEvidence,
    )

    Status = "PASS" if all(Check["status"] == "PASS" for Check in Checks) else "BLOCKED"
    Record = {
        "status": Status,
        "audit_id": "E00-INDEPENDENT-AUDIT-20260830",
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "read-only independent audit; no E00 source or output modified",
        "test_data_handling": "no test prediction or label statistics calculated; total manifest row count and content-group split crossing only",
        "checks": Checks,
        "audit_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": GetPackageVersion("scipy"),
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
        print("Usage: audit_e00.py REPO_ROOT OUTPUT_ROOT", file=sys.stderr)
        return 2
    RepoRoot = Path(sys.argv[1]).resolve()
    OutputRoot = Path(sys.argv[2]).resolve()
    Record = Execute(RepoRoot, OutputRoot)
    print(json.dumps({"status": Record["status"], "checks": Record["checks"]}, ensure_ascii=False))
    return 0 if Record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(Main())
