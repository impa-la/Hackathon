# /// <summary>
# Visible CMD entry point for resumable E01-R5 full three-seed training
# /// </summary>

from __future__ import annotations

import json
from pathlib import Path

from .full_training import RunFullTrainingResumable, ValidateFixedR5Contract


def Main() -> int:
    SourceRoot = Path(__file__).resolve().parent
    DeepvoiceRoot = SourceRoot.parents[1]
    ConfigPath = SourceRoot / "config.json"
    Config = json.loads(ConfigPath.read_text(encoding="utf-8"))
    Config["resolved_cache_root"] = str(
        DeepvoiceRoot / Config["cache_relative_path"]
    )
    ValidateFixedR5Contract(Config)
    print("E01-R5 guarded FP32 full training", flush=True)
    print(
        "workload=32,768 samples/epoch x 20 epochs x 3 seeds; "
        "batch=32; workers=2; progress every 25 batches",
        flush=True,
    )
    print(
        "Ctrl-C is safe after the current operation returns; relaunch the same "
        "command to resume from the latest completed epoch.",
        flush=True,
    )
    try:
        RunFullTrainingResumable(
            DeepvoiceRoot,
            SourceRoot,
            ConfigPath,
            Config,
        )
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(Main())
