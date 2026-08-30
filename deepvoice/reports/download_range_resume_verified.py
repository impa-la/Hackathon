from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


RANGE_PATTERN = re.compile(r"^(\d+)-(\d+)\.part$")
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume one large file with verified parallel byte ranges."
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--run-json", type=Path, required=True)
    return parser.parse_args()


def session() -> requests.Session:
    value = getattr(THREAD_LOCAL, "session", None)
    if value is None:
        value = requests.Session()
        value.headers.update({"User-Agent": "DeepVoice-data-audit/1.0"})
        THREAD_LOCAL.session = value
    return value


def parse_part(path: Path) -> tuple[int, int] | None:
    match = RANGE_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def prepare_part_directory(part_dir: Path, destination_bytes: int) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    ranges = [parse_part(path) for path in part_dir.glob("*.part")]
    ranges = [item for item in ranges if item is not None]
    if ranges and destination_bytes != min(start for start, _ in ranges):
        for path in part_dir.glob("*.part"):
            path.unlink()


def download_range(
    url: str,
    part_dir: Path,
    start: int,
    end: int,
    retries: int = 6,
) -> Path:
    path = part_dir / f"{start}-{end}.part"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return path
    for attempt in range(1, retries + 1):
        existing = path.stat().st_size if path.exists() else 0
        if existing > expected:
            path.unlink()
            existing = 0
        request_start = start + existing
        headers = {"Range": f"bytes={request_start}-{end}"}
        try:
            with session().get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
                expected_content_range = f"bytes {request_start}-{end}/"
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(expected_content_range):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")
                with path.open("ab" if existing else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if path.stat().st_size != expected:
                raise RuntimeError(
                    f"range size mismatch {path.stat().st_size} != {expected}"
                )
            return path
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def file_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("--workers must be between 1 and 12")
    if args.chunk_mib < 32:
        raise ValueError("--chunk-mib must be at least 32")
    if not args.destination.is_file():
        raise FileNotFoundError(args.destination)
    initial_bytes = args.destination.stat().st_size
    if initial_bytes > args.total_bytes:
        raise RuntimeError("destination is larger than the official file")

    started = time.monotonic()
    part_dir = args.destination.parent / f"{args.destination.name}.ranges"
    prepare_part_directory(part_dir, initial_bytes)
    chunk_bytes = args.chunk_mib * 1024 * 1024
    ranges: list[tuple[int, int]] = []
    start = initial_bytes
    while start < args.total_bytes:
        end = min(args.total_bytes - 1, start + chunk_bytes - 1)
        ranges.append((start, end))
        start = end + 1

    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_range, args.url, part_dir, start, end): (start, end)
            for start, end in ranges
        }
        for index, future in enumerate(as_completed(futures), start=1):
            start, end = futures[future]
            future.result()
            completed_bytes += end - start + 1
            print(
                f"[{index:03d}/{len(ranges):03d}] ranges complete; "
                f"{completed_bytes / (1024**3):.2f} GiB received",
                flush=True,
            )

    with args.destination.open("ab") as destination:
        for index, (start, end) in enumerate(ranges, start=1):
            path = part_dir / f"{start}-{end}.part"
            expected = end - start + 1
            if path.stat().st_size != expected:
                raise RuntimeError(f"cannot assemble incomplete range: {path}")
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
            path.unlink()
            print(f"[{index:03d}/{len(ranges):03d}] ranges assembled", flush=True)
    part_dir.rmdir()

    if args.destination.stat().st_size != args.total_bytes:
        raise RuntimeError("assembled file size does not match official size")
    observed_md5, observed_sha256 = file_hashes(args.destination)
    if observed_md5.lower() != args.expected_md5.lower():
        raise RuntimeError(
            f"MD5 mismatch: {observed_md5} != {args.expected_md5.lower()}"
        )
    run = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "destination": str(args.destination),
        "initial_bytes": initial_bytes,
        "final_bytes": args.destination.stat().st_size,
        "workers": args.workers,
        "chunk_mib": args.chunk_mib,
        "range_count": len(ranges),
        "md5": observed_md5,
        "sha256": observed_sha256,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.run_json.parent.mkdir(parents=True, exist_ok=True)
    args.run_json.write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
