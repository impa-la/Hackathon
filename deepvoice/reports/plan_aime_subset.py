from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


APPROVED_OPEN_MODEL_PROVIDERS = (
    "AudioLDM 2 Large",
    "AudioLDM 2 Music",
    "MusicGen Large",
    "MusicGen Medium",
    "MusicGen Small",
    "Mustango",
    "Riffusion",
    "Stable Audio v1",
    "Stable Audio v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a provider-balanced AIME pilot subset from audited shards."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards-per-provider", type=int, default=4)
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count < 1 or count > length:
        raise ValueError("Requested shard count is outside the candidate range")
    if count == 1:
        return [length // 2]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def main() -> None:
    args = parse_args()
    with args.inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        inventory = list(csv.DictReader(handle))

    selected: list[dict[str, str | int]] = []
    for provider in APPROVED_OPEN_MODEL_PROVIDERS:
        candidates = []
        for row in inventory:
            models = json.loads(row["models_json"])
            if list(models) != [provider] or int(row["row_count"]) < 30:
                continue
            candidates.append(row)
        candidates.sort(key=lambda row: int(row["first_id"]))
        indices = evenly_spaced_indices(len(candidates), args.shards_per_provider)
        for ordinal, candidate_index in enumerate(indices, start=1):
            row = candidates[candidate_index]
            selected.append(
                {
                    "provider": provider,
                    "provider_shard_ordinal": ordinal,
                    "repository_path": row["path"],
                    "target_filename": Path(row["path"]).name,
                    "expected_bytes": int(row["size_bytes"]),
                    "expected_sha256": row["lfs_sha256"],
                    "expected_rows": int(row["row_count"]),
                    "first_id": row["first_id"],
                    "last_id": row["last_id"],
                    "selection_reason": "pure_provider_evenly_spaced_id_range",
                    "download_status": "PENDING",
                }
            )

    if len(selected) != len(APPROVED_OPEN_MODEL_PROVIDERS) * args.shards_per_provider:
        raise RuntimeError("The subset plan did not produce the expected number of shards")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(
        json.dumps(
            {
                "providers": len(APPROVED_OPEN_MODEL_PROVIDERS),
                "shards": len(selected),
                "expected_rows": sum(int(row["expected_rows"]) for row in selected),
                "expected_bytes": sum(int(row["expected_bytes"]) for row in selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
