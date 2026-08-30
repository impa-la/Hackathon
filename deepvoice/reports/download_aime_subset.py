from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


DATASET = "disco-eth/AIME"
RESOLVE_ROOT = "https://huggingface.co/datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify a pinned AIME subset.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_bytes: int) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if existing and response.status_code == 200:
            existing = 0
            part.unlink(missing_ok=True)
        elif existing and response.status_code != 206:
            response.raise_for_status()
        else:
            response.raise_for_status()
        mode = "ab" if existing else "wb"
        with part.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if part.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"size mismatch for {destination.name}: {part.stat().st_size} != {expected_bytes}"
        )
    os.replace(part, destination)


def main() -> None:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    args.result_csv.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.plan.open("r", encoding="utf-8-sig", newline="") as handle:
        plan = list(csv.DictReader(handle))

    results: list[dict[str, str | int]] = []
    for index, row in enumerate(plan, start=1):
        target = args.destination / row["target_filename"]
        expected_bytes = int(row["expected_bytes"])
        expected_hash = row["expected_sha256"].lower()
        status = "DOWNLOADED"
        if target.is_file() and target.stat().st_size == expected_bytes:
            observed_hash = sha256_file(target)
            if observed_hash == expected_hash:
                status = "VERIFIED_EXISTING"
            else:
                raise RuntimeError(f"existing hash mismatch: {target}")
        else:
            url = (
                f"{RESOLVE_ROOT}/{DATASET}/resolve/{args.revision}/"
                f"{quote(row['repository_path'])}?download=true"
            )
            download(url, target, expected_bytes)
            observed_hash = sha256_file(target)
            if observed_hash != expected_hash:
                raise RuntimeError(f"downloaded hash mismatch: {target}")
        result = dict(row)
        result.update(
            {
                "download_status": status,
                "observed_bytes": target.stat().st_size,
                "observed_sha256": observed_hash,
                "local_path": str(target),
            }
        )
        results.append(result)
        print(
            f"[{index:02d}/{len(plan):02d}] {status} "
            f"{row['provider']} {target.name}",
            flush=True,
        )

    fields = list(results[0])
    with args.result_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    run = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "revision": args.revision,
        "files": len(results),
        "rows": sum(int(row["expected_rows"]) for row in results),
        "bytes": sum(int(row["observed_bytes"]) for row in results),
        "elapsed_seconds": time.monotonic() - started,
        "hash_mismatches": 0,
    }
    args.run_json.write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
