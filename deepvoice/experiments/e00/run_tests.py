# /// <summary>
# Dependency-light executable checks for the isolated DeepVoice E00 contract
# /// </summary>

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ModuleRoot = Path(__file__).resolve().parent
if str(ModuleRoot.parent.parent) not in sys.path:
    sys.path.insert(0, str(ModuleRoot.parent.parent))

from experiments.e00.contract import (  # noqa: E402
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
