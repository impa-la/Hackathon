# /// <summary>
# Test-isolated manifest records for E01 training and validation
# /// </summary>

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract_adapter import (
    AuditGroupCrossings,
    BuildLabelMasks,
    HashFile,
    LoadManifestPartitions,
    ProjectCrossingRows,
    ValidateNonTestManifestRows,
)


ExpectedManifestSha256 = (
    "2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6"
)


@dataclass(frozen=True)
class AudioRecord:
    Dataset: str
    SampleId: str
    SourceFamily: str
    GeneratorOrProvider: str
    ContentGroupKey: str
    Split: str
    Locator: str
    Codec: str
    SampleRateHz: str
    Channels: str
    DurationSeconds: float
    Labels: tuple[float, ...]
    Masks: tuple[bool, ...]

    def ToMetricRow(self) -> dict[str, str]:
        return {
            "dataset": self.Dataset,
            "sample_id": self.SampleId,
            "source_family": self.SourceFamily,
            "generator_or_provider": self.GeneratorOrProvider,
            "content_group_key": self.ContentGroupKey,
            "recommended_content_split": self.Split,
            "codec": self.Codec,
            "sample_rate_hz": self.SampleRateHz,
            "channels": self.Channels,
            "duration_seconds": str(self.DurationSeconds),
        }


def ConvertRowsToRecords(
    Rows: list[dict[str, str]],
    Labels: np.ndarray,
    Masks: np.ndarray,
) -> list[AudioRecord]:
    Records = []
    for RowIndex, Row in enumerate(Rows):
        Records.append(
            AudioRecord(
                Dataset=Row["dataset"],
                SampleId=Row["sample_id"],
                SourceFamily=Row["source_family"],
                GeneratorOrProvider=Row["generator_or_provider"],
                ContentGroupKey=Row["content_group_key"],
                Split=Row["recommended_content_split"],
                Locator=Row["source_locator"],
                Codec=Row["codec"],
                SampleRateHz=Row["sample_rate_hz"],
                Channels=Row["channels"],
                DurationSeconds=float(Row["duration_seconds"]),
                Labels=tuple(float(Value) for Value in Labels[RowIndex]),
                Masks=tuple(bool(Value) for Value in Masks[RowIndex]),
            )
        )
    return Records


def LoadE01Records(
    ManifestPath: Path,
) -> tuple[list[AudioRecord], list[AudioRecord], dict[str, Any]]:
    ManifestSha256 = HashFile(ManifestPath)
    if ManifestSha256 != ExpectedManifestSha256:
        raise RuntimeError(
            f"Manifest SHA mismatch: expected {ExpectedManifestSha256}, found {ManifestSha256}"
        )

    NonTestRows, TestCrossingRows, FieldNames, TotalRowCount = LoadManifestPartitions(
        ManifestPath
    )
    ValidateNonTestManifestRows(NonTestRows, FieldNames)
    CrossingRows = ProjectCrossingRows(NonTestRows) + TestCrossingRows
    Crossings, CrossingSummary = AuditGroupCrossings(CrossingRows)
    if Crossings:
        raise RuntimeError(f"Content groups cross fixed splits: {len(Crossings)}")

    Labels, Masks = BuildLabelMasks(NonTestRows)
    Records = ConvertRowsToRecords(NonTestRows, Labels, Masks)
    TrainingRecords = [Record for Record in Records if Record.Split == "train"]
    ValidationRecords = [Record for Record in Records if Record.Split == "validation"]
    Summary = {
        "manifest_sha256": ManifestSha256,
        "manifest_total_row_count": TotalRowCount,
        "crossing_group_count": CrossingSummary["crossing_group_count"],
        "train_row_count": len(TrainingRecords),
        "validation_row_count": len(ValidationRecords),
        "test_field_contract": {
            "allowed_fields": [
                "content_group_key",
                "recommended_content_split",
            ],
            "retained_forbidden_fields": 0,
            "test_statistics": 0,
        },
    }
    return TrainingRecords, ValidationRecords, Summary
