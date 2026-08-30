# /// <summary>
# Read-only resource and raw-locator preflight for the E01 full-run gate
# /// </summary>

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch

from .audio import LoadLocatorWaveform, ParseLocator
from .contract_adapter import (
    ExpectedE00ContractSha256,
    ObservedE00ContractSha256,
)
from .records import AudioRecord, LoadE01Records


def GetPackageVersion(PackageName: str) -> str | None:
    try:
        return importlib.metadata.version(PackageName)
    except importlib.metadata.PackageNotFoundError:
        return None


def RunCommand(Command: list[str]) -> dict[str, Any]:
    try:
        Result = subprocess.run(
            Command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return {
            "available": True,
            "return_code": Result.returncode,
            "stdout": Result.stdout.strip(),
            "stderr": Result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as Error:
        return {
            "available": False,
            "error": repr(Error),
        }


def ProbeLocator(Record: AudioRecord, Decode: bool) -> dict[str, Any]:
    Parsed = ParseLocator(Record.Locator)
    Result: dict[str, Any] = {
        "scope": "first_non_test_record_for_dataset",
        "dataset": Record.Dataset,
        "kind": Parsed.Kind,
        "container_exists": Parsed.ContainerPath.is_file(),
        "container_bytes": (
            Parsed.ContainerPath.stat().st_size if Parsed.ContainerPath.is_file() else None
        ),
        "member_exists": None,
        "decode_requested": Decode,
        "decode_status": "NOT_REQUESTED",
    }
    if Parsed.Kind == "zip" and Parsed.ContainerPath.is_file():
        if Parsed.Member is None:
            raise AssertionError("ZIP member missing")
        with zipfile.ZipFile(Parsed.ContainerPath, "r") as ZipFile:
            try:
                ZipFile.getinfo(Parsed.Member)
                Result["member_exists"] = True
            except KeyError:
                Result["member_exists"] = False
    if Decode:
        try:
            Started = time.perf_counter()
            Waveform = LoadLocatorWaveform(Record.Locator, 16000)
            Result["decode_status"] = "PASS"
            Result["decoded_sample_count"] = int(Waveform.numel())
            Result["decode_seconds"] = time.perf_counter() - Started
        except Exception as Error:
            Result["decode_status"] = "BLOCKED"
            Result["decode_error"] = f"{type(Error).__name__}: {Error}"
    return Result


def RunPreflight(DeepvoiceRoot: Path, Config: dict[str, Any]) -> dict[str, Any]:
    ManifestPath = DeepvoiceRoot / Config["manifest_relative_path"]
    TrainingRecords, ValidationRecords, ManifestSummary = LoadE01Records(ManifestPath)
    AuditPath = DeepvoiceRoot / Config["e00_r2_audit_relative_path"]
    AuditFirstLine = (
        AuditPath.read_text(encoding="utf-8").splitlines()[0]
        if AuditPath.is_file()
        else "MISSING"
    )
    PackageVersions = {
        Name: GetPackageVersion(Name)
        for Name in ("torch", "torchaudio", "pyarrow", "numpy", "scipy")
    }
    Ffmpeg = RunCommand(["ffmpeg", "-version"])
    Ffprobe = RunCommand(["ffprobe", "-version"])
    NvidiaSmi = RunCommand(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    PythonLaunchers = RunCommand(["py", "-0p"])
    DiskUsage = shutil.disk_usage(DeepvoiceRoot)
    InDedicatedEnvironment = ".venv-e01" in str(Path(sys.executable)).casefold()
    CudaReady = bool(torch.cuda.is_available())
    TorchVersionReady = str(torch.__version__).startswith("2.7.1+cu126")
    TorchaudioReady = PackageVersions["torchaudio"] == "2.7.1+cu126"
    PyarrowReady = PackageVersions["pyarrow"] == "25.0.1"
    FfmpegReady = bool(Ffmpeg.get("available") and Ffmpeg.get("return_code") == 0)
    DiskReady = DiskUsage.free >= int(
        float(Config["full_run_requires"]["minimum_free_disk_gib"]) * 1024**3
    )
    AuditReady = AuditFirstLine == Config["e00_r2_audit_required_first_line"]
    ContractReady = ObservedE00ContractSha256 == ExpectedE00ContractSha256

    FirstByDataset: dict[str, AudioRecord] = {}
    for Record in [*TrainingRecords, *ValidationRecords]:
        FirstByDataset.setdefault(Record.Dataset, Record)
    LocatorProbes = []
    for Dataset in sorted(FirstByDataset):
        Decode = Dataset != "aime-open-model-subset" or PyarrowReady
        LocatorProbes.append(ProbeLocator(FirstByDataset[Dataset], Decode))
    LocatorContainersReady = all(
        Probe["container_exists"]
        and Probe.get("member_exists") is not False
        for Probe in LocatorProbes
    )
    RequiredDecodeReady = all(
        Probe["decode_status"] == "PASS" for Probe in LocatorProbes
    )

    Gates = {
        "dedicated_venv": InDedicatedEnvironment,
        "cuda_available": CudaReady,
        "torch_2_7_1_cu126": TorchVersionReady,
        "torchaudio_2_7_1_cu126": TorchaudioReady,
        "pyarrow_25_0_1": PyarrowReady,
        "ffmpeg": FfmpegReady,
        "free_disk_at_least_12_gib": DiskReady,
        "e00_r2_audit_pass": AuditReady,
        "e00_r2_contract_hash": ContractReady,
        "raw_locator_containers": LocatorContainersReady,
        "all_four_locator_decoders": RequiredDecodeReady,
    }
    Blockers = [GateName for GateName, Passed in Gates.items() if not Passed]
    return {
        "status": "READY" if not Blockers else "BLOCKED_RESOURCE",
        "blockers": Blockers,
        "gates": Gates,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "packages": PackageVersions,
            "torch_cuda_available": CudaReady,
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_device_count": torch.cuda.device_count(),
            "physical_gpu": NvidiaSmi,
            "python_launchers": PythonLaunchers,
            "ffmpeg": Ffmpeg,
            "ffprobe": Ffprobe,
            "disk_total_bytes": DiskUsage.total,
            "disk_free_bytes": DiskUsage.free,
            "disk_free_gib": DiskUsage.free / 1024**3,
        },
        "manifest": ManifestSummary,
        "locator_probes": LocatorProbes,
        "installation_commands_windows": Config["installation_commands_windows"],
        "expected_install_disk_gib_with_cache": Config[
            "expected_install_disk_gib_with_cache"
        ],
        "minimum_free_disk_gib_after_install": Config["full_run_requires"][
            "minimum_free_disk_gib"
        ],
        "install_sources": Config["official_install_sources"],
    }


def Main() -> int:
    DeepvoiceRoot = Path(__file__).resolve().parents[2]
    ConfigPath = Path(__file__).resolve().parent / "config.json"
    Config = json.loads(ConfigPath.read_text(encoding="utf-8"))
    Result = RunPreflight(DeepvoiceRoot, Config)
    print(json.dumps(Result, indent=2, ensure_ascii=False))
    return 0 if Result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(Main())
