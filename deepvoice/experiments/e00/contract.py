# /// <summary>
# Standalone DeepVoice E00 label, metric, split, bootstrap and shortcut contract
# /// </summary>

from __future__ import annotations

import csv
import gzip
import hashlib
import math
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
DetectionHeadIndices = (0, 1, 2)
PresenceHeadIndices = (3, 4)
ExpectedManifestSha256 = (
    "2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6"
)
RequiredManifestColumns = (
    "dataset",
    "label",
    "sample_id",
    "source_family",
    "generator_or_provider",
    "content_group_key",
    "recommended_content_split",
    "provider_holdout_group",
    "codec",
    "sample_rate_hz",
    "channels",
    "duration_seconds",
    "training_eligible",
)


def HashFile(FilePath: Path) -> str:
    Digest = hashlib.sha256()
    with FilePath.open("rb") as FileHandle:
        while True:
            Chunk = FileHandle.read(1024 * 1024)
            if not Chunk:
                break
            Digest.update(Chunk)
    return Digest.hexdigest()


def LoadManifestRows(ManifestPath: Path) -> tuple[list[dict[str, str]], list[str]]:
    with gzip.open(ManifestPath, "rt", encoding="utf-8-sig", newline="") as FileHandle:
        Reader = csv.DictReader(FileHandle)
        if Reader.fieldnames is None:
            raise ValueError("Manifest header is missing")
        Rows = list(Reader)
        FieldNames = list(Reader.fieldnames)
    return Rows, FieldNames


def ValidateManifestRows(Rows: Sequence[dict[str, str]], FieldNames: Sequence[str]) -> None:
    MissingColumns = sorted(set(RequiredManifestColumns) - set(FieldNames))
    if MissingColumns:
        raise ValueError(f"Manifest columns are missing: {MissingColumns}")
    if not Rows:
        raise ValueError("Manifest contains no rows")

    AllowedDatasets = {
        "ljspeech-1.1",
        "wavefake-1.2.0",
        "fma-small",
        "aime-open-model-subset",
    }
    AllowedSplits = {"train", "validation", "test"}
    SeenKeys: set[tuple[str, str]] = set()
    for Row in Rows:
        Dataset = Row["dataset"]
        Split = Row["recommended_content_split"]
        RowKey = (Dataset, Row["sample_id"])
        if Dataset not in AllowedDatasets:
            raise ValueError(f"Unsupported dataset in final manifest: {Dataset}")
        if Split not in AllowedSplits:
            raise ValueError(f"Unsupported recommended split: {Split}")
        if Row["training_eligible"].casefold() != "true":
            raise ValueError(f"Ineligible row is present in final manifest: {RowKey}")
        if not Row["content_group_key"]:
            raise ValueError(f"Missing content group: {RowKey}")
        if RowKey in SeenKeys:
            raise ValueError(f"Duplicate dataset/sample key: {RowKey}")
        SeenKeys.add(RowKey)


def BuildLabelMasks(
    Rows: Sequence[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    RowCount = len(Rows)
    Labels = np.full((RowCount, len(HeadNames)), np.nan, dtype=np.float64)
    Masks = np.zeros((RowCount, len(HeadNames)), dtype=bool)
    DatasetContract = {
        "ljspeech-1.1": ((0.0, 0.0, np.nan, 1.0, 0.0), "real"),
        "wavefake-1.2.0": ((1.0, 1.0, np.nan, 1.0, 0.0), "synthetic"),
        "fma-small": ((0.0, np.nan, 0.0, np.nan, 1.0), "real"),
        "aime-open-model-subset": ((1.0, np.nan, 1.0, 0.0, 1.0), "synthetic"),
    }

    for RowIndex, Row in enumerate(Rows):
        Dataset = Row["dataset"]
        if Dataset not in DatasetContract:
            raise ValueError(f"No label contract exists for dataset: {Dataset}")
        Values, ExpectedLabel = DatasetContract[Dataset]
        if Row["label"] != ExpectedLabel:
            raise ValueError(
                f"Manifest label mismatch for {Dataset}/{Row['sample_id']}: "
                f"expected {ExpectedLabel}, found {Row['label']}"
            )
        ValueArray = np.asarray(Values, dtype=np.float64)
        LabelMask = np.isfinite(ValueArray)
        Labels[RowIndex, LabelMask] = ValueArray[LabelMask]
        Masks[RowIndex, LabelMask] = True

    return Labels, Masks


def ValidateBinaryTargets(Targets: np.ndarray, MetricName: str) -> None:
    if Targets.ndim != 1:
        raise ValueError(f"{MetricName} targets must be one-dimensional")
    if not np.isfinite(Targets).all():
        raise ValueError(f"{MetricName} targets must be finite")
    UniqueValues = np.unique(Targets)
    if not np.all(np.isin(UniqueValues, (0.0, 1.0))):
        raise ValueError(f"{MetricName} targets must be binary")
    if UniqueValues.size != 2:
        raise ValueError(f"{MetricName} needs both target classes")


def ValidateScores(Targets: np.ndarray, Scores: np.ndarray, MetricName: str) -> None:
    if Targets.shape != Scores.shape or Scores.ndim != 1:
        raise ValueError(f"{MetricName} targets and scores must have equal 1D shapes")
    if not np.isfinite(Scores).all():
        raise ValueError(f"{MetricName} scores must be finite")
    if np.any((Scores < 0.0) | (Scores > 1.0)):
        raise ValueError(f"{MetricName} scores must be in [0, 1]")
    ValidateBinaryTargets(Targets, MetricName)


def CalculateEer(Targets: Iterable[float], Scores: Iterable[float]) -> float:
    TargetArray = np.asarray(list(Targets), dtype=np.float64)
    ScoreArray = np.asarray(list(Scores), dtype=np.float64)
    ValidateScores(TargetArray, ScoreArray, "EER")

    Order = np.argsort(-ScoreArray, kind="mergesort")
    SortedTargets = TargetArray[Order]
    SortedScores = ScoreArray[Order]
    DistinctIndices = np.where(np.diff(SortedScores))[0]
    ThresholdIndices = np.r_[DistinctIndices, SortedTargets.size - 1]
    TruePositives = np.cumsum(SortedTargets)[ThresholdIndices]
    FalsePositives = 1 + ThresholdIndices - TruePositives
    PositiveCount = TargetArray.sum()
    NegativeCount = TargetArray.size - PositiveCount
    FalsePositiveRate = np.r_[0.0, FalsePositives / NegativeCount]
    TruePositiveRate = np.r_[0.0, TruePositives / PositiveCount]
    FalseNegativeRate = 1.0 - TruePositiveRate
    EqualIndex = int(np.argmin(np.abs(FalsePositiveRate - FalseNegativeRate)))
    return float(
        (FalsePositiveRate[EqualIndex] + FalseNegativeRate[EqualIndex]) / 2.0
    )


def CalculateRocAuc(Targets: Iterable[float], Scores: Iterable[float]) -> float:
    TargetArray = np.asarray(list(Targets), dtype=np.float64)
    ScoreArray = np.asarray(list(Scores), dtype=np.float64)
    ValidateScores(TargetArray, ScoreArray, "AUC")

    Order = np.argsort(ScoreArray, kind="mergesort")
    SortedScores = ScoreArray[Order]
    Ranks = np.empty(ScoreArray.size, dtype=np.float64)
    Start = 0
    while Start < ScoreArray.size:
        End = Start + 1
        while End < ScoreArray.size and SortedScores[End] == SortedScores[Start]:
            End += 1
        AverageRank = (Start + 1 + End) / 2.0
        Ranks[Order[Start:End]] = AverageRank
        Start = End

    PositiveCount = int(TargetArray.sum())
    NegativeCount = TargetArray.size - PositiveCount
    PositiveRankSum = Ranks[TargetArray == 1.0].sum()
    Area = PositiveRankSum - PositiveCount * (PositiveCount + 1) / 2.0
    return float(Area / (PositiveCount * NegativeCount))


def CalculateBrier(Targets: Iterable[float], Scores: Iterable[float]) -> float:
    TargetArray = np.asarray(list(Targets), dtype=np.float64)
    ScoreArray = np.asarray(list(Scores), dtype=np.float64)
    ValidateScores(TargetArray, ScoreArray, "Brier")
    return float(np.mean(np.square(ScoreArray - TargetArray)))


def CalculateLogLoss(Targets: Iterable[float], Scores: Iterable[float]) -> float:
    TargetArray = np.asarray(list(Targets), dtype=np.float64)
    ScoreArray = np.asarray(list(Scores), dtype=np.float64)
    ValidateScores(TargetArray, ScoreArray, "LogLoss")
    ClippedScores = np.clip(ScoreArray, 1e-12, 1.0 - 1e-12)
    Losses = -(
        TargetArray * np.log(ClippedScores)
        + (1.0 - TargetArray) * np.log(1.0 - ClippedScores)
    )
    return float(np.mean(Losses))


def CalculateExpectedCalibrationError(
    Targets: Iterable[float],
    Scores: Iterable[float],
    BinCount: int = 15,
) -> float:
    TargetArray = np.asarray(list(Targets), dtype=np.float64)
    ScoreArray = np.asarray(list(Scores), dtype=np.float64)
    ValidateScores(TargetArray, ScoreArray, "ECE")
    if BinCount <= 0:
        raise ValueError("ECE bin count must be positive")

    Edges = np.linspace(0.0, 1.0, BinCount + 1)
    BinIndices = np.minimum(np.searchsorted(Edges, ScoreArray, side="right") - 1, BinCount - 1)
    CalibrationError = 0.0
    for BinIndex in range(BinCount):
        BinMask = BinIndices == BinIndex
        if not np.any(BinMask):
            continue
        Confidence = float(np.mean(ScoreArray[BinMask]))
        Accuracy = float(np.mean(TargetArray[BinMask]))
        CalibrationError += float(np.mean(BinMask)) * abs(Confidence - Accuracy)
    return float(CalibrationError)


def CalculateHeadMetrics(
    Labels: np.ndarray,
    Masks: np.ndarray,
    Predictions: np.ndarray,
) -> list[dict[str, Any]]:
    ExpectedShape = (Labels.shape[0], len(HeadNames))
    if Labels.shape != ExpectedShape or Masks.shape != ExpectedShape:
        raise ValueError(f"Labels and masks must have shape {ExpectedShape}")
    if Predictions.shape != ExpectedShape:
        raise ValueError(f"Predictions must have shape {ExpectedShape}")
    if not np.isfinite(Predictions).all():
        raise ValueError("Predictions must be finite")
    if np.any((Predictions < 0.0) | (Predictions > 1.0)):
        raise ValueError("Predictions must be in [0, 1]")

    MetricRows: list[dict[str, Any]] = []
    for HeadIndex, HeadName in enumerate(HeadNames):
        ObservedMask = Masks[:, HeadIndex]
        Targets = Labels[ObservedMask, HeadIndex]
        Scores = Predictions[ObservedMask, HeadIndex]
        UniqueCount = np.unique(Targets).size
        MetricRow: dict[str, Any] = {
            "head": HeadName,
            "weight": float(HeadWeights[HeadIndex]),
            "observed_count": int(Targets.size),
            "positive_count": int(Targets.sum()) if Targets.size else 0,
            "status": "OK" if UniqueCount == 2 else "INSUFFICIENT_CLASSES",
            "auc": None,
            "eer": None,
            "brier": None,
            "log_loss": None,
            "ece_15": None,
            "selection_component": None,
        }
        if UniqueCount == 2:
            MetricRow["auc"] = CalculateRocAuc(Targets, Scores)
            MetricRow["eer"] = CalculateEer(Targets, Scores)
            MetricRow["brier"] = CalculateBrier(Targets, Scores)
            MetricRow["log_loss"] = CalculateLogLoss(Targets, Scores)
            MetricRow["ece_15"] = CalculateExpectedCalibrationError(Targets, Scores)
            if HeadIndex in DetectionHeadIndices:
                MetricRow["selection_component"] = 1.0 - MetricRow["eer"]
            else:
                MetricRow["selection_component"] = MetricRow["auc"]
        MetricRows.append(MetricRow)
    return MetricRows


def CalculateCompetitionProxy(HeadMetrics: Sequence[dict[str, Any]]) -> dict[str, float]:
    if len(HeadMetrics) != len(HeadNames):
        raise ValueError("Five head metric rows are required")
    Components = []
    for HeadIndex, MetricRow in enumerate(HeadMetrics):
        if MetricRow["head"] != HeadNames[HeadIndex]:
            raise ValueError("Head metric order does not match the official contract")
        if MetricRow["selection_component"] is None:
            raise ValueError(f"Official proxy cannot be calculated for {MetricRow['head']}")
        Components.append(float(MetricRow["selection_component"]))

    Ads = 0.5 * Components[0] + 0.2 * Components[1] + 0.3 * Components[2]
    Cps = 0.5 * Components[3] + 0.5 * Components[4]
    Score = 0.9 * Ads + 0.1 * Cps
    EffectiveScore = float(np.dot(HeadWeights, np.asarray(Components)))
    if abs(Score - EffectiveScore) > 1e-12:
        raise AssertionError("Expanded official weights do not match ADS/CPS score")
    return {
        "ADS": float(Ads),
        "CPS": float(Cps),
        "OfficialValidationProxy": float(Score),
        "RobustSelectionScore": float(Score),
    }


def BuildFixturePredictions(RowCount: int, Seed: int) -> np.ndarray:
    if RowCount <= 0:
        raise ValueError("Fixture row count must be positive")
    Generator = np.random.default_rng(Seed)
    return Generator.uniform(0.025, 0.975, size=(RowCount, len(HeadNames)))


def AuditGroupCrossings(
    Rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    GroupSplits: dict[str, set[str]] = defaultdict(set)
    GroupRowCounts: dict[str, int] = defaultdict(int)
    SplitCounts: dict[str, int] = defaultdict(int)
    for Row in Rows:
        GroupKey = Row["content_group_key"]
        Split = Row["recommended_content_split"]
        GroupSplits[GroupKey].add(Split)
        GroupRowCounts[GroupKey] += 1
        SplitCounts[Split] += 1

    Crossings = []
    for GroupKey in sorted(GroupSplits):
        Splits = sorted(GroupSplits[GroupKey])
        if len(Splits) > 1:
            Crossings.append(
                {
                    "content_group_key": GroupKey,
                    "splits": "|".join(Splits),
                    "row_count": GroupRowCounts[GroupKey],
                }
            )
    Summary = {
        "row_count": len(Rows),
        "content_group_count": len(GroupSplits),
        "crossing_group_count": len(Crossings),
        "train_row_count": SplitCounts.get("train", 0),
        "validation_row_count": SplitCounts.get("validation", 0),
        "test_row_count": SplitCounts.get("test", 0),
    }
    return Crossings, Summary


def BootstrapByContentGroup(
    Rows: Sequence[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
    Predictions: np.ndarray,
    Seed: int,
    Replicates: int,
    Confidence: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if Replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    if not 0.0 < Confidence < 1.0:
        raise ValueError("Bootstrap confidence must be in (0, 1)")

    GroupIndices: dict[str, list[int]] = defaultdict(list)
    for RowIndex, Row in enumerate(Rows):
        GroupIndices[Row["content_group_key"]].append(RowIndex)
    GroupKeys = sorted(GroupIndices)
    Generator = np.random.default_rng(Seed)
    ReplicateRows: list[dict[str, float]] = []

    for ReplicateIndex in range(Replicates):
        SampledPositions = Generator.integers(0, len(GroupKeys), size=len(GroupKeys))
        RowIndexParts = [GroupIndices[GroupKeys[int(Position)]] for Position in SampledPositions]
        RowIndices = np.fromiter(
            (Index for Part in RowIndexParts for Index in Part),
            dtype=np.int64,
        )
        Metrics = CalculateHeadMetrics(
            Labels[RowIndices],
            Masks[RowIndices],
            Predictions[RowIndices],
        )
        if any(MetricRow["status"] != "OK" for MetricRow in Metrics):
            continue
        Proxy = CalculateCompetitionProxy(Metrics)
        ResultRow = {
            "replicate": float(ReplicateIndex),
            "RobustSelectionScore": Proxy["RobustSelectionScore"],
        }
        for MetricRow in Metrics:
            ResultRow[f"{MetricRow['head']}_component"] = float(
                MetricRow["selection_component"]
            )
        ReplicateRows.append(ResultRow)

    if not ReplicateRows:
        raise RuntimeError("No valid content-group bootstrap replicate was produced")

    LowerQuantile = (1.0 - Confidence) / 2.0
    UpperQuantile = 1.0 - LowerQuantile
    Summary: dict[str, Any] = {
        "seed": Seed,
        "requested_replicates": Replicates,
        "valid_replicates": len(ReplicateRows),
        "confidence": Confidence,
        "group_count": len(GroupKeys),
        "intervals": {},
    }
    MetricNames = [Name for Name in ReplicateRows[0] if Name != "replicate"]
    for MetricName in MetricNames:
        Values = np.asarray([Row[MetricName] for Row in ReplicateRows], dtype=np.float64)
        Summary["intervals"][MetricName] = {
            "mean": float(np.mean(Values)),
            "lower": float(np.quantile(Values, LowerQuantile)),
            "upper": float(np.quantile(Values, UpperQuantile)),
        }
    return ReplicateRows, Summary


def AggregateSegmentProbabilities(SegmentPredictions: np.ndarray) -> np.ndarray:
    SegmentArray = np.asarray(SegmentPredictions, dtype=np.float64)
    if SegmentArray.ndim != 2 or SegmentArray.shape[1] != len(HeadNames):
        raise ValueError(f"Segments must have shape (N, {len(HeadNames)})")
    if SegmentArray.shape[0] == 0:
        raise ValueError("At least one segment is required")
    if not np.isfinite(SegmentArray).all():
        raise ValueError("Segment probabilities must be finite")
    if np.any((SegmentArray < 0.0) | (SegmentArray > 1.0)):
        raise ValueError("Segment probabilities must be in [0, 1]")
    return np.mean(SegmentArray, axis=0, dtype=np.float64)


def EvaluateSingletonEquivalence(
    Seed: int,
    FileCount: int = 64,
    MaximumSegments: int = 8,
) -> dict[str, Any]:
    if FileCount <= 0 or MaximumSegments <= 0:
        raise ValueError("File and segment counts must be positive")
    Generator = np.random.default_rng(Seed)
    FileSegments = []
    for _ in range(FileCount):
        SegmentCount = int(Generator.integers(1, MaximumSegments + 1))
        FileSegments.append(
            Generator.uniform(0.0, 1.0, size=(SegmentCount, len(HeadNames)))
        )

    SingletonOutputs = np.stack(
        [AggregateSegmentProbabilities(Segments) for Segments in FileSegments]
    )
    BatchedOutputs = np.empty_like(SingletonOutputs)
    BatchOrder = Generator.permutation(FileCount)
    for FileIndex in BatchOrder:
        BatchedOutputs[FileIndex] = AggregateSegmentProbabilities(FileSegments[FileIndex])

    MaxAbsoluteDelta = float(np.max(np.abs(SingletonOutputs - BatchedOutputs)))
    OtherFilePermutationDelta = 0.0
    AnchorOutput = AggregateSegmentProbabilities(FileSegments[0])
    for _ in range(10):
        Generator.shuffle(BatchOrder)
        CurrentOutput = AggregateSegmentProbabilities(FileSegments[0])
        OtherFilePermutationDelta = max(
            OtherFilePermutationDelta,
            float(np.max(np.abs(AnchorOutput - CurrentOutput))),
        )
    return {
        "seed": Seed,
        "file_count": FileCount,
        "maximum_segments": MaximumSegments,
        "max_absolute_delta": MaxAbsoluteDelta,
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


def BuildShortcutLabelAudit(
    Rows: Sequence[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
    Scope: str,
) -> list[dict[str, Any]]:
    AxisGetters = {
        "dataset": lambda Row: Row["dataset"],
        "source_family": lambda Row: Row["source_family"],
        "generator_or_provider": lambda Row: Row["generator_or_provider"],
        "codec": lambda Row: Row["codec"],
        "sample_rate_hz": lambda Row: Row["sample_rate_hz"],
        "channels": lambda Row: Row["channels"],
        "duration_bucket_seconds": lambda Row: MakeDurationBucket(Row["duration_seconds"]),
    }
    AuditRows: list[dict[str, Any]] = []
    for AxisName, Getter in AxisGetters.items():
        SliceIndices: dict[str, list[int]] = defaultdict(list)
        for RowIndex, Row in enumerate(Rows):
            SliceIndices[str(Getter(Row))].append(RowIndex)
        for SliceValue in sorted(SliceIndices):
            Indices = np.asarray(SliceIndices[SliceValue], dtype=np.int64)
            FileTargets = Labels[Indices, 0]
            FileFakeRate = float(np.mean(FileTargets))
            LabelPure = bool(FileFakeRate == 0.0 or FileFakeRate == 1.0)
            Result: dict[str, Any] = {
                "scope": Scope,
                "axis": AxisName,
                "slice": SliceValue,
                "row_count": int(Indices.size),
                "file_fake_rate": FileFakeRate,
                "file_label_pure": LabelPure,
                "shortcut_risk": "HIGH" if LabelPure else "MONITOR",
            }
            for HeadIndex, HeadName in enumerate(HeadNames):
                HeadMask = Masks[Indices, HeadIndex]
                ObservedTargets = Labels[Indices, HeadIndex][HeadMask]
                Result[f"{HeadName}_observed_count"] = int(ObservedTargets.size)
                Result[f"{HeadName}_positive_rate"] = (
                    float(np.mean(ObservedTargets)) if ObservedTargets.size else None
                )
            AuditRows.append(Result)
    return AuditRows


def BuildShortcutMetricAudit(
    Rows: Sequence[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
    Predictions: np.ndarray,
    Scope: str,
) -> list[dict[str, Any]]:
    AxisGetters = {
        "dataset": lambda Row: Row["dataset"],
        "source_family": lambda Row: Row["source_family"],
        "generator_or_provider": lambda Row: Row["generator_or_provider"],
        "codec": lambda Row: Row["codec"],
        "sample_rate_hz": lambda Row: Row["sample_rate_hz"],
        "channels": lambda Row: Row["channels"],
        "duration_bucket_seconds": lambda Row: MakeDurationBucket(Row["duration_seconds"]),
    }
    AuditRows: list[dict[str, Any]] = []
    for AxisName, Getter in AxisGetters.items():
        SliceIndices: dict[str, list[int]] = defaultdict(list)
        for RowIndex, Row in enumerate(Rows):
            SliceIndices[str(Getter(Row))].append(RowIndex)
        for SliceValue in sorted(SliceIndices):
            Indices = np.asarray(SliceIndices[SliceValue], dtype=np.int64)
            Metrics = CalculateHeadMetrics(
                Labels[Indices],
                Masks[Indices],
                Predictions[Indices],
            )
            for MetricRow in Metrics:
                AuditRows.append(
                    {
                        "scope": Scope,
                        "axis": AxisName,
                        "slice": SliceValue,
                        **MetricRow,
                    }
                )
    return AuditRows


def SummarizeLabelMasks(
    Rows: Sequence[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
) -> list[dict[str, Any]]:
    DatasetIndices: dict[str, list[int]] = defaultdict(list)
    for RowIndex, Row in enumerate(Rows):
        DatasetIndices[Row["dataset"]].append(RowIndex)
    SummaryRows: list[dict[str, Any]] = []
    for Dataset in sorted(DatasetIndices):
        Indices = np.asarray(DatasetIndices[Dataset], dtype=np.int64)
        for HeadIndex, HeadName in enumerate(HeadNames):
            HeadMask = Masks[Indices, HeadIndex]
            ObservedTargets = Labels[Indices, HeadIndex][HeadMask]
            UniqueTargets = "|".join(
                str(int(Value)) for Value in np.unique(ObservedTargets)
            )
            SummaryRows.append(
                {
                    "dataset": Dataset,
                    "head": HeadName,
                    "row_count": int(Indices.size),
                    "observed_count": int(ObservedTargets.size),
                    "masked_count": int(Indices.size - ObservedTargets.size),
                    "observed_values": UniqueTargets,
                }
            )
    return SummaryRows


def IsClose(Left: float, Right: float, Tolerance: float = 1e-12) -> bool:
    return math.isclose(Left, Right, rel_tol=0.0, abs_tol=Tolerance)
