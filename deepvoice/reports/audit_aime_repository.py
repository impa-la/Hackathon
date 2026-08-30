from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import polars as pl
import requests


DATASET = "disco-eth/AIME"
API_ROOT = "https://huggingface.co/api/datasets"
RESOLVE_ROOT = "https://huggingface.co/datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit AIME repository metadata using projected HTTP range reads."
    )
    parser.add_argument(
        "--revision",
        default="b84d4be5eda830b6eb714998569dba73530f2601",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def get_json(url: str) -> tuple[Any, requests.Response]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json(), response


def list_tree(revision: str) -> list[dict[str, Any]]:
    url = (
        f"{API_ROOT}/{DATASET}/tree/{revision}/data"
        "?recursive=true&expand=true&limit=100"
    )
    items: list[dict[str, Any]] = []
    while url:
        page, response = get_json(url)
        items.extend(item for item in page if item.get("type") == "file")
        url = response.links.get("next", {}).get("url", "")
    return sorted(items, key=lambda item: item["path"])


def audit_shard(item: dict[str, Any], revision: str) -> dict[str, Any]:
    path = item["path"]
    url = f"{RESOLVE_ROOT}/{DATASET}/resolve/{revision}/{quote(path)}"
    frame = pl.read_parquet(url, columns=["id", "model"])
    model_counts = Counter(frame.get_column("model").to_list())
    ids = frame.get_column("id").to_list()
    lfs = item.get("lfs") or {}
    return {
        "path": path,
        "size_bytes": int(item["size"]),
        "lfs_sha256": lfs.get("oid", ""),
        "row_count": frame.height,
        "first_id": ids[0] if ids else "",
        "last_id": ids[-1] if ids else "",
        "model_count": len(model_counts),
        "models_json": json.dumps(dict(sorted(model_counts.items())), ensure_ascii=False),
        "error": "",
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be between 1 and 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repository, _ = get_json(f"{API_ROOT}/{DATASET}/revision/{args.revision}")
    if repository.get("sha") != args.revision:
        raise RuntimeError("Pinned revision did not resolve to itself")
    if repository.get("private") or repository.get("gated"):
        raise RuntimeError("AIME is no longer public and ungated")

    items = list_tree(args.revision)
    if not items:
        raise RuntimeError("No parquet shards found")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_shard, item, args.revision): item for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # keep a complete failure inventory
                lfs = item.get("lfs") or {}
                rows.append(
                    {
                        "path": item["path"],
                        "size_bytes": int(item["size"]),
                        "lfs_sha256": lfs.get("oid", ""),
                        "row_count": 0,
                        "first_id": "",
                        "last_id": "",
                        "model_count": 0,
                        "models_json": "{}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    rows.sort(key=lambda row: row["path"])

    inventory_path = args.output_dir / "aime-shard-inventory.csv"
    with inventory_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    providers: Counter[str] = Counter()
    for row in rows:
        providers.update(json.loads(row["models_json"]))
    summary = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "revision": args.revision,
        "last_modified": repository.get("lastModified"),
        "private": repository.get("private"),
        "gated": repository.get("gated"),
        "shard_count": len(rows),
        "failed_shards": sum(bool(row["error"]) for row in rows),
        "published_bytes": sum(row["size_bytes"] for row in rows),
        "projected_rows": sum(row["row_count"] for row in rows),
        "provider_counts": dict(sorted(providers.items())),
        "read_scope": ["id", "model"],
        "note": "HTTP range projection; audio bytes were not downloaded by this audit.",
    }
    run_path = args.output_dir / "aime-repository-audit-run.json"
    run_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
