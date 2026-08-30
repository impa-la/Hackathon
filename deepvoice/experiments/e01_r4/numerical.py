# /// <summary>
# Hard numerical integrity guards for quality-first FP32 E01 optimization
# /// </summary>

from __future__ import annotations

from typing import Any

import torch


class NumericalIntegrityError(RuntimeError):
    def __init__(self, Evidence: dict[str, Any]) -> None:
        self.Evidence = Evidence
        super().__init__(
            "Numerical integrity failure at "
            f"{Evidence['stage']} batch {Evidence['batch_index']}: "
            f"{Evidence['tensor_name']}"
        )


def TensorFiniteEvidence(
    Tensor: torch.Tensor,
    Stage: str,
    BatchIndex: int,
    TensorName: str,
) -> dict[str, Any]:
    Detached = Tensor.detach()
    FiniteMask = torch.isfinite(Detached)
    NonFiniteCount = int((~FiniteMask).sum().cpu())
    Evidence: dict[str, Any] = {
        "stage": Stage,
        "batch_index": BatchIndex,
        "tensor_name": TensorName,
        "shape": list(Detached.shape),
        "dtype": str(Detached.dtype),
        "device": str(Detached.device),
        "nonfinite_count": NonFiniteCount,
    }
    if NonFiniteCount:
        Indices = torch.nonzero(~FiniteMask, as_tuple=False)[:8].cpu().tolist()
        Evidence["first_nonfinite_indices"] = Indices
        FiniteValues = Detached[FiniteMask]
        if FiniteValues.numel():
            Evidence["finite_min"] = float(FiniteValues.min().cpu())
            Evidence["finite_max"] = float(FiniteValues.max().cpu())
    return Evidence


def RequireFiniteTensor(
    Tensor: torch.Tensor,
    Stage: str,
    BatchIndex: int,
    TensorName: str,
) -> None:
    Evidence = TensorFiniteEvidence(Tensor, Stage, BatchIndex, TensorName)
    if Evidence["nonfinite_count"]:
        raise NumericalIntegrityError(Evidence)


def RequireMaskedLabels(
    Labels: torch.Tensor,
    Masks: torch.Tensor,
    Stage: str,
    BatchIndex: int,
) -> None:
    if Labels.shape != Masks.shape:
        raise NumericalIntegrityError(
            {
                "stage": Stage,
                "batch_index": BatchIndex,
                "tensor_name": "labels_masks_shape",
                "labels_shape": list(Labels.shape),
                "masks_shape": list(Masks.shape),
            }
        )
    ObservedLabels = Labels[Masks]
    RequireFiniteTensor(ObservedLabels, Stage, BatchIndex, "observed_labels")
    if not torch.all((ObservedLabels == 0.0) | (ObservedLabels == 1.0)):
        raise NumericalIntegrityError(
            {
                "stage": Stage,
                "batch_index": BatchIndex,
                "tensor_name": "observed_labels_binary_contract",
                "nonbinary_count": int(
                    (~((ObservedLabels == 0.0) | (ObservedLabels == 1.0))).sum().cpu()
                ),
            }
        )


def RequireModelParametersFinite(
    Model: torch.nn.Module,
    Stage: str,
    BatchIndex: int,
    Suffix: str,
) -> None:
    for Name, Parameter in Model.named_parameters():
        RequireFiniteTensor(
            Parameter,
            Stage,
            BatchIndex,
            f"parameter_{Suffix}:{Name}",
        )


def RequireGradientsFinite(
    Model: torch.nn.Module,
    Stage: str,
    BatchIndex: int,
) -> int:
    GradientCount = 0
    for Name, Parameter in Model.named_parameters():
        if Parameter.grad is None:
            continue
        GradientCount += 1
        RequireFiniteTensor(
            Parameter.grad,
            Stage,
            BatchIndex,
            f"unscaled_gradient:{Name}",
        )
    if GradientCount == 0:
        raise NumericalIntegrityError(
            {
                "stage": Stage,
                "batch_index": BatchIndex,
                "tensor_name": "unscaled_gradients",
                "gradient_tensor_count": 0,
            }
        )
    return GradientCount


def OptimizerStepValue(Optimizer: torch.optim.Optimizer) -> int:
    Values = []
    for State in Optimizer.state.values():
        Step = State.get("step")
        if Step is None:
            continue
        Values.append(int(Step.item()) if torch.is_tensor(Step) else int(Step))
    return min(Values) if Values else 0


def RequireFp32TrainingMode(Config: dict[str, Any]) -> dict[str, Any]:
    PrecisionMode = Config.get("precision_mode")
    if PrecisionMode != "fp32_guarded":
        raise RuntimeError(
            "E01-R4 forbids AMP/GradScaler training; precision_mode must be "
            "fp32_guarded"
        )
    return {
        "precision_mode": "fp32_guarded",
        "autocast_enabled": False,
        "grad_scaler_enabled": False,
    }


def GuardedFp32OptimizationStep(
    Model: torch.nn.Module,
    Optimizer: torch.optim.Optimizer,
    Logits: torch.Tensor,
    Loss: torch.Tensor,
    Stage: str,
    BatchIndex: int,
    MaximumGradientNorm: float = 5.0,
) -> dict[str, Any]:
    if torch.is_autocast_enabled():
        raise NumericalIntegrityError(
            {
                "stage": Stage,
                "batch_index": BatchIndex,
                "tensor_name": "autocast_forbidden",
                "autocast_enabled": True,
            }
        )
    if Logits.dtype != torch.float32 or Loss.dtype != torch.float32:
        raise NumericalIntegrityError(
            {
                "stage": Stage,
                "batch_index": BatchIndex,
                "tensor_name": "fp32_dtype_contract",
                "logits_dtype": str(Logits.dtype),
                "loss_dtype": str(Loss.dtype),
            }
        )
    RequireFiniteTensor(Logits, Stage, BatchIndex, "logits")
    RequireFiniteTensor(Loss, Stage, BatchIndex, "loss")
    RequireModelParametersFinite(Model, Stage, BatchIndex, "before_step")
    StepBefore = OptimizerStepValue(Optimizer)
    Loss.backward()
    GradientTensorCount = RequireGradientsFinite(Model, Stage, BatchIndex)
    GradientNorm = torch.nn.utils.clip_grad_norm_(Model.parameters(), MaximumGradientNorm)
    RequireFiniteTensor(GradientNorm, Stage, BatchIndex, "gradient_norm")
    Optimizer.step()
    StepAfter = OptimizerStepValue(Optimizer)
    if StepAfter != StepBefore + 1:
        raise NumericalIntegrityError(
            {
                "stage": Stage,
                "batch_index": BatchIndex,
                "tensor_name": "optimizer_step_behavior",
                "optimizer_step_before": StepBefore,
                "optimizer_step_after": StepAfter,
            }
        )
    RequireModelParametersFinite(Model, Stage, BatchIndex, "after_step")
    return {
        "precision_mode": "fp32_guarded",
        "autocast_enabled": False,
        "grad_scaler_enabled": False,
        "optimizer_step_before": StepBefore,
        "optimizer_step_after": StepAfter,
        "gradient_tensor_count": GradientTensorCount,
        "gradient_norm": float(GradientNorm.detach().cpu()),
    }
