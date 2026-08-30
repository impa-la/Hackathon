# /// <summary>
# Dependency-light executable checks for the isolated DeepVoice E00 contract
# /// </summary>

from __future__ import annotations

import json
import csv
import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np


ModuleRoot = Path(__file__).resolve().parent
if str(ModuleRoot.parent.parent) not in sys.path:
    sys.path.insert(0, str(ModuleRoot.parent.parent))

from experiments.e00_r2.contract import (  # noqa: E402
    AuditGroupCrossings,
    BootstrapByContentGroup,
    BuildFixturePredictions,
    BuildLabelMasks,
    CalculateBrier,
    CalculateCompetitionProxy,
    CalculateEer,
    CalculateHeadMetrics,
    CalculateRocAuc,
    EvaluateSingletonEquivalence,
    HeadNames,
    IsClose,
    LoadManifestPartitions,
    ProjectCrossingRows,
    RequiredManifestColumns,
    ValidateNonTestManifestRows,
)


def CheckOfficialMetricExample() -> None:
    Targets = np.asarray((0.0, 0.0, 1.0, 1.0))
    Scores = np.asarray((0.1, 0.4, 0.35, 0.8))
    assert IsClose(CalculateEer(Targets, Scores), 0.5)
    assert IsClose(CalculateRocAuc(Targets, Scores), 0.75)
    assert IsClose(CalculateBrier(Targets, Scores), 0.158125)


def CheckOfficialWeightExpansion() -> None:
    Components = (0.9, 0.8, 0.7, 0.8, 0.6)
    Metrics = []
    for HeadIndex, HeadName in enumerate(HeadNames):
        Metrics.append(
            {
                "head": HeadName,
                "selection_component": Components[HeadIndex],
            }
        )
    Proxy = CalculateCompetitionProxy(Metrics)
    assert IsClose(Proxy["ADS"], 0.82)
    assert IsClose(Proxy["CPS"], 0.7)
    assert IsClose(Proxy["OfficialValidationProxy"], 0.808)
    assert IsClose(Proxy["RobustSelectionScore"], 0.808)


def CheckLabelMaskMapping() -> None:
    Rows = [
        {"dataset": "ljspeech-1.1", "label": "real", "sample_id": "ljs"},
        {"dataset": "wavefake-1.2.0", "label": "synthetic", "sample_id": "wave"},
        {"dataset": "fma-small", "label": "real", "sample_id": "fma"},
        {"dataset": "aime-open-model-subset", "label": "synthetic", "sample_id": "aime"},
    ]
    Labels, Masks = BuildLabelMasks(Rows)
    ExpectedMasks = np.asarray(
        (
            (1, 1, 0, 1, 1),
            (1, 1, 0, 1, 1),
            (1, 0, 1, 0, 1),
            (1, 0, 1, 1, 1),
        ),
        dtype=bool,
    )
    ExpectedObservedLabels = np.asarray(
        (0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1),
        dtype=np.float64,
    )
    assert np.array_equal(Masks, ExpectedMasks)
    assert np.array_equal(Labels[Masks], ExpectedObservedLabels)


def CheckGroupCrossingAudit() -> None:
    Rows = [
        {"content_group_key": "a", "recommended_content_split": "train"},
        {"content_group_key": "a", "recommended_content_split": "validation"},
        {"content_group_key": "b", "recommended_content_split": "test"},
    ]
    Crossings, Summary = AuditGroupCrossings(Rows)
    assert Summary["crossing_group_count"] == 1
    assert Crossings[0]["content_group_key"] == "a"


def MakeManifestRow(
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
    return {Column: Values[Column] for Column in RequiredManifestColumns}


def WriteSentinelManifest(ManifestPath: Path, TestSentinel: str) -> None:
    TestRow = MakeManifestRow(
        TestSentinel,
        TestSentinel,
        TestSentinel,
        "test",
        "test-group",
    )
    for Column in RequiredManifestColumns:
        if Column not in ("content_group_key", "recommended_content_split"):
            TestRow[Column] = f"{TestSentinel}:{Column}"
    Rows = [
        MakeManifestRow("ljspeech-1.1", "real", "ljs", "validation", "ljs"),
        MakeManifestRow(
            "wavefake-1.2.0",
            "synthetic",
            "wave",
            "validation",
            "wave",
        ),
        MakeManifestRow("fma-small", "real", "fma", "validation", "fma"),
        MakeManifestRow(
            "aime-open-model-subset",
            "synthetic",
            "aime",
            "validation",
            "aime",
        ),
        TestRow,
    ]
    with gzip.open(ManifestPath, "wt", encoding="utf-8", newline="") as FileHandle:
        Writer = csv.DictWriter(FileHandle, fieldnames=RequiredManifestColumns)
        Writer.writeheader()
        Writer.writerows(Rows)


def CheckTestSentinelIsolation() -> None:
    with tempfile.TemporaryDirectory() as TemporaryDirectory:
        FirstPath = Path(TemporaryDirectory) / "first.csv.gz"
        SecondPath = Path(TemporaryDirectory) / "second.csv.gz"
        WriteSentinelManifest(FirstPath, "SENTINEL_INVALID_A")
        WriteSentinelManifest(SecondPath, "SENTINEL_INVALID_B")
        FirstNonTest, FirstTest, FirstFields, FirstTotal = LoadManifestPartitions(
            FirstPath
        )
        SecondNonTest, SecondTest, SecondFields, SecondTotal = LoadManifestPartitions(
            SecondPath
        )
        assert FirstTotal == SecondTotal == 5
        assert FirstTest == SecondTest == [
            {
                "content_group_key": "test-group",
                "recommended_content_split": "test",
            }
        ]
        ValidateNonTestManifestRows(FirstNonTest, FirstFields)
        ValidateNonTestManifestRows(SecondNonTest, SecondFields)
        FirstLabels, FirstMasks = BuildLabelMasks(FirstNonTest)
        SecondLabels, SecondMasks = BuildLabelMasks(SecondNonTest)
        Predictions = BuildFixturePredictions(len(FirstNonTest), 20260830)
        FirstMetrics = CalculateHeadMetrics(FirstLabels, FirstMasks, Predictions)
        SecondMetrics = CalculateHeadMetrics(SecondLabels, SecondMasks, Predictions)
        assert FirstMetrics == SecondMetrics
        assert CalculateCompetitionProxy(FirstMetrics) == CalculateCompetitionProxy(
            SecondMetrics
        )
        FirstCrossings = AuditGroupCrossings(
            ProjectCrossingRows(FirstNonTest) + FirstTest
        )
        SecondCrossings = AuditGroupCrossings(
            ProjectCrossingRows(SecondNonTest) + SecondTest
        )
        assert FirstCrossings == SecondCrossings


def CheckCrossingAuditRejectsMetadata() -> None:
    try:
        AuditGroupCrossings(
            [
                {
                    "content_group_key": "group",
                    "recommended_content_split": "test",
                    "label": "forbidden",
                }
            ]
        )
    except ValueError:
        return
    raise AssertionError("Crossing audit accepted forbidden test metadata")


def CheckContentGroupBootstrapReproducibility() -> None:
    Rows = []
    Labels = []
    Masks = []
    for RowIndex in range(40):
        Target = float(RowIndex % 2)
        Rows.append({"content_group_key": f"group-{RowIndex:02d}"})
        Labels.append((Target, Target, Target, Target, Target))
        Masks.append((1, 1, 1, 1, 1))
    LabelArray = np.asarray(Labels, dtype=np.float64)
    MaskArray = np.asarray(Masks, dtype=bool)
    Predictions = BuildFixturePredictions(len(Rows), 123)
    FirstRows, FirstSummary = BootstrapByContentGroup(
        Rows,
        LabelArray,
        MaskArray,
        Predictions,
        Seed=456,
        Replicates=25,
        Confidence=0.95,
    )
    SecondRows, SecondSummary = BootstrapByContentGroup(
        Rows,
        LabelArray,
        MaskArray,
        Predictions,
        Seed=456,
        Replicates=25,
        Confidence=0.95,
    )
    assert FirstRows == SecondRows
    assert FirstSummary == SecondSummary


def CheckSingletonEquivalence() -> None:
    Result = EvaluateSingletonEquivalence(20260830)
    assert Result["max_absolute_delta"] <= 1e-6
    assert Result["other_file_permutation_delta"] <= 1e-6


def CheckMaskAwareMetrics() -> None:
    Labels = np.asarray(
        (
            (0, 0, np.nan, 1, 0),
            (1, 1, np.nan, 1, 0),
            (0, np.nan, 0, np.nan, 1),
            (1, np.nan, 1, 0, 1),
        ),
        dtype=np.float64,
    )
    Masks = np.isfinite(Labels)
    Predictions = np.asarray(
        (
            (0.1, 0.1, 0.3, 0.9, 0.1),
            (0.9, 0.9, 0.7, 0.8, 0.2),
            (0.2, 0.4, 0.1, 0.7, 0.8),
            (0.8, 0.5, 0.9, 0.1, 0.9),
        ),
        dtype=np.float64,
    )
    Metrics = CalculateHeadMetrics(Labels, Masks, Predictions)
    assert all(Row["status"] == "OK" for Row in Metrics)
    assert [Row["observed_count"] for Row in Metrics] == [4, 2, 2, 3, 4]
    assert IsClose(CalculateCompetitionProxy(Metrics)["OfficialValidationProxy"], 1.0)


def RunAllTests() -> dict[str, object]:
    Checks = (
        CheckOfficialMetricExample,
        CheckOfficialWeightExpansion,
        CheckLabelMaskMapping,
        CheckGroupCrossingAudit,
        CheckTestSentinelIsolation,
        CheckCrossingAuditRejectsMetadata,
        CheckContentGroupBootstrapReproducibility,
        CheckSingletonEquivalence,
        CheckMaskAwareMetrics,
    )
    Results = []
    for Check in Checks:
        Check()
        Results.append({"check": Check.__name__, "status": "PASS"})
    return {"status": "PASS", "check_count": len(Results), "checks": Results}


if __name__ == "__main__":
    print(json.dumps(RunAllTests(), indent=2, ensure_ascii=False))
