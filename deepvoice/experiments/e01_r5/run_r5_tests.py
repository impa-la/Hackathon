# /// <summary>
# R4 regression plus R5 progress, resume identity and strict numerical tests
# /// </summary>

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from .determinism import ConfigureParentCpuThreads
from .full_training import (
    LoadEpochResume,
    PrintTrainProgress,
    RequireNestedTensorsFinite,
    ResumePaths,
    SaveEpochResume,
    ValidateFixedR5Contract,
)
from .numerical import GuardedFp32OptimizationStep, NumericalIntegrityError
from .resume import (
    AtomicWriteJson,
    FormatDuration,
    HashFile,
    ProgressMath,
    StrictLoadJson,
    ValidateRunIdentity,
)
from .run_tests import RunAllTests, RunTinySmoke
from .strict_serialization import JsonBytes, NonFinitePayloadError


def CheckProgressAndEtaMath() -> None:
    Result = ProgressMath(
        CompletedUnits=250,
        TotalUnits=1000,
        IntervalCompletedUnits=50,
        IntervalElapsedSeconds=2.0,
        RunElapsedSeconds=5.0,
    )
    assert Result == {
        "overall_percent": 25.0,
        "units_per_second": 25.0,
        "elapsed_seconds": 5.0,
        "eta_seconds": 30.0,
    }
    assert FormatDuration(30.0) == "00:00:30"
    Buffer = io.StringIO()
    with contextlib.redirect_stdout(Buffer):
        PrintTrainProgress(
            20260830,
            0,
            3,
            0,
            20,
            24,
            1024,
            0.5,
            0.6,
            Result,
            {"cuda_allocated_gib": 0.25, "cuda_reserved_gib": 0.5},
        )
    Text = Buffer.getvalue()
    for Required in (
        "seed=20260830",
        "epoch=1/20",
        "batch=25/1024",
        "overall=25.00%",
        "loss=0.500000",
        "mean_loss=0.600000",
        "samples/s=25.00",
        "elapsed=00:00:05",
        "ETA=00:00:30",
        "CUDA_alloc=0.250GiB",
        "CUDA_reserved=0.500GiB",
    ):
        assert Required in Text


def CheckResumeHashRejection() -> None:
    Expected = {
        "identity_schema": "e01-r5-run-identity-v1",
        "config_sha256": "a" * 64,
        "code_inventory_sha256": "b" * 64,
        "cache_index_sha256": "c" * 64,
    }
    ValidateRunIdentity(Expected, dict(Expected))
    Observed = dict(Expected)
    Observed["code_inventory_sha256"] = "d" * 64
    try:
        ValidateRunIdentity(Expected, Observed)
    except RuntimeError as Error:
        assert "code_inventory_sha256" in str(Error)
        return
    raise AssertionError("Changed code hash was accepted for resume")


def CheckAuthorizingR4AuditPin(Config: dict[str, Any]) -> None:
    assert Config["e01_r4_audit_relative_path"] == (
        "reports/e01-r4-validation-audit.md"
    )
    assert Config["e01_r4_audit_sha256"] == (
        "1aac0423b4e655e349bcbdb47f4cbce0a7df24eeabcf9c4638905778b9ea9203"
    )
    assert Config["e01_r4_audit_required_first_line"] == "EXPERIMENT_AUDIT: PASS"
    DeepvoiceRoot = Path(__file__).resolve().parents[2]
    AuditPath = DeepvoiceRoot / Config["e01_r4_audit_relative_path"]
    assert AuditPath.is_file()
    assert HashFile(AuditPath) == Config["e01_r4_audit_sha256"]
    assert AuditPath.read_text(encoding="utf-8").splitlines()[0] == (
        Config["e01_r4_audit_required_first_line"]
    )


def CheckR4CommonFilesByteIdentical() -> None:
    ExperimentsRoot = Path(__file__).resolve().parent.parent
    R4Root = ExperimentsRoot / "e01_r4"
    R5Root = ExperimentsRoot / "e01_r5"
    CommonFiles = (
        "__init__.py",
        "audio.py",
        "benchmark.py",
        "cache.py",
        "contract_adapter.py",
        "determinism.py",
        "model.py",
        "numerical.py",
        "preflight.py",
        "records.py",
        "run_e01.py",
        "run_tests.py",
        "sampling.py",
        "strict_serialization.py",
        "train_e01.py",
    )
    for FileName in CommonFiles:
        assert HashFile(R4Root / FileName) == HashFile(R5Root / FileName), FileName


def CheckAtomicEpochResumeRoundTripAndHashRejection() -> None:
    Identity = {
        "identity_schema": "e01-r5-run-identity-v1",
        "config_sha256": "a" * 64,
        "code_inventory_sha256": "b" * 64,
        "cache_index_sha256": "c" * 64,
    }
    Config = {"epochs": 20, "fixture": "atomic-round-trip"}
    Numerical = {
        "guarded_batch_count": 1,
        "maximum_gradient_norm": 0.5,
        "optimizer_skip_count": 0,
    }
    EpochRows = [
        {
            "epoch": 1,
            "mean_train_loss": 0.5,
            "samples": 32,
            "batches": 1,
            "seconds": 1.0,
        }
    ]
    with tempfile.TemporaryDirectory() as TemporaryDirectory:
        ArtifactsRoot = Path(TemporaryDirectory)
        Model = torch.nn.Linear(2, 1)
        Optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-3)
        Pointer = SaveEpochResume(
            ArtifactsRoot,
            20260830,
            1,
            EpochRows,
            Numerical,
            Model,  # type: ignore[arg-type]
            Optimizer,
            Config,
            Identity,
        )
        assert Pointer["next_epoch"] == 1
        assert not list(ArtifactsRoot.rglob("*.tmp-*"))
        RestoredModel = torch.nn.Linear(2, 1)
        RestoredOptimizer = torch.optim.AdamW(RestoredModel.parameters(), lr=1e-3)
        NextEpoch, RestoredRows, RestoredNumerical = LoadEpochResume(
            ArtifactsRoot,
            20260830,
            RestoredModel,  # type: ignore[arg-type]
            RestoredOptimizer,
            Config,
            Identity,
        )
        assert NextEpoch == 1
        assert RestoredRows == EpochRows
        assert RestoredNumerical == Numerical
        for ExpectedParameter, ObservedParameter in zip(
            Model.parameters(), RestoredModel.parameters()
        ):
            assert torch.equal(ExpectedParameter, ObservedParameter)
        _, LatestPath = ResumePaths(ArtifactsRoot, 20260830)
        CorruptedPointer = StrictLoadJson(LatestPath)
        CorruptedPointer["run_identity"]["config_sha256"] = "d" * 64
        AtomicWriteJson(LatestPath, CorruptedPointer)
        try:
            LoadEpochResume(
                ArtifactsRoot,
                20260830,
                RestoredModel,  # type: ignore[arg-type]
                RestoredOptimizer,
                Config,
                Identity,
            )
        except RuntimeError as Error:
            assert "config_sha256" in str(Error)
            return
        raise AssertionError("Resume pointer with changed config hash was accepted")


def CheckStrictJsonAndNestedTensorGuards() -> None:
    try:
        JsonBytes({"outer": {"loss": float("nan")}})
    except NonFinitePayloadError:
        pass
    else:
        raise AssertionError("Nested NaN escaped strict JSON")
    try:
        RequireNestedTensorsFinite({"optimizer": torch.tensor([1.0, float("nan")])})
    except NumericalIntegrityError as Error:
        assert "optimizer" in Error.Evidence["tensor_name"]
        return
    raise AssertionError("Nonfinite optimizer tensor escaped resume checkpoint guard")


def CheckGuardedFp32StepAndSkippedStep() -> None:
    Model = torch.nn.Linear(2, 1)
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-3)
    Inputs = torch.ones((2, 2))
    Logits = Model(Inputs)
    Loss = Logits.square().mean()
    Optimizer.zero_grad(set_to_none=True)
    Evidence = GuardedFp32OptimizationStep(
        Model,
        Optimizer,
        Logits,
        Loss,
        "r5_fp32_test",
        0,
    )
    assert Evidence["precision_mode"] == "fp32_guarded"
    assert Evidence["optimizer_step_before"] == 0
    assert Evidence["optimizer_step_after"] == 1


def RunR5Checks(Config: dict[str, Any]) -> dict[str, Any]:
    ValidateFixedR5Contract(Config)
    Checks: tuple[Callable[[], None], ...] = (
        CheckProgressAndEtaMath,
        CheckResumeHashRejection,
        lambda: CheckAuthorizingR4AuditPin(Config),
        CheckR4CommonFilesByteIdentical,
        CheckAtomicEpochResumeRoundTripAndHashRejection,
        CheckStrictJsonAndNestedTensorGuards,
        CheckGuardedFp32StepAndSkippedStep,
    )
    Rows = []
    for Check in Checks:
        Check()
        Rows.append({"check": Check.__name__, "status": "PASS"})
    return {"status": "PASS", "check_count": len(Rows), "checks": Rows}


def Main() -> int:
    ConfigPath = Path(__file__).resolve().parent / "config.json"
    Config = json.loads(ConfigPath.read_text(encoding="utf-8"))
    ConfigureParentCpuThreads(Config)
    Base = RunAllTests(Config)
    R5 = RunR5Checks(Config)
    Smoke = RunTinySmoke(Config)
    Payload = {
        "status": "PASS",
        "base_r4_regression": Base,
        "r5_resume_progress": R5,
        "cpu_smoke": Smoke,
        "total_check_count": Base["check_count"] + R5["check_count"],
    }
    print(JsonBytes(Payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
