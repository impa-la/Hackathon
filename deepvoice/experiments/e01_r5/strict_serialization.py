# /// <summary>
# Strict JSON serialization that rejects every nonfinite nested scalar
# /// </summary>

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


class NonFinitePayloadError(ValueError):
    pass


def AssertFinitePayload(Value: Any, Path: str = "$") -> None:
    if isinstance(Value, bool) or Value is None or isinstance(Value, (str, int)):
        return
    if isinstance(Value, float):
        if not math.isfinite(Value):
            raise NonFinitePayloadError(f"Nonfinite JSON scalar at {Path}: {Value!r}")
        return
    if isinstance(Value, Mapping):
        for Key, NestedValue in Value.items():
            AssertFinitePayload(NestedValue, f"{Path}.{Key}")
        return
    if isinstance(Value, Sequence) and not isinstance(Value, (bytes, bytearray)):
        for Index, NestedValue in enumerate(Value):
            AssertFinitePayload(NestedValue, f"{Path}[{Index}]")
        return


def JsonBytes(Value: Any) -> bytes:
    AssertFinitePayload(Value)
    return (
        json.dumps(
            Value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def JsonLine(Value: Any) -> str:
    AssertFinitePayload(Value)
    return json.dumps(
        Value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

