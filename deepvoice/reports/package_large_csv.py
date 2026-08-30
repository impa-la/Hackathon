#!/usr/bin/env python3
"""Create deterministic gzip copies of large DeepVoice CSV evidence files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


DefaultNames = [
    "wavefake-audio-inventory.csv",
    "wavefake-source-group-manifest.csv",
    "wavefake-duplicate-groups.csv",
    "wavefake-ljspeech-pairing.csv",
    "deepvoice-training-manifest.csv",
]


def Main() -> None:
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--directory", type=Path, required=True)
    Parser.add_argument("--name", action="append", dest="names")
    Args = Parser.parse_args()
    Directory = Args.directory.resolve()
    Results = []
    for Name in Args.names or DefaultNames:
        Source = Directory / Name
        Target = Source.with_suffix(Source.suffix + ".gz")
        with Source.open("rb") as Input, Target.open("wb") as RawOutput:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=RawOutput,
                compresslevel=9,
                mtime=0,
            ) as Output:
                while True:
                    Block = Input.read(1024 * 1024)
                    if not Block:
                        break
                    Output.write(Block)
        Results.append(
            {
                "source": Source.name,
                "source_bytes": Source.stat().st_size,
                "source_sha256": hashlib.sha256(Source.read_bytes()).hexdigest(),
                "gzip": Target.name,
                "gzip_bytes": Target.stat().st_size,
                "gzip_sha256": hashlib.sha256(Target.read_bytes()).hexdigest(),
                "gzip_mtime": 0,
            }
        )
    (Directory / "large-csv-package-run.json").write_text(
        json.dumps({"files": Results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"files": Results}, indent=2))


if __name__ == "__main__":
    Main()
