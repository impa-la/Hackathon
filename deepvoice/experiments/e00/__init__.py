# /// <summary>
# Public entry points for the isolated DeepVoice E00 evaluation contract
# /// </summary>

from .contract import (
    BuildFixturePredictions,
    BuildLabelMasks,
    CalculateCompetitionProxy,
    CalculateEer,
    CalculateHeadMetrics,
    CalculateRocAuc,
    EvaluateSingletonEquivalence,
    HeadNames,
    HeadWeights,
)

__all__ = (
    "BuildFixturePredictions",
    "BuildLabelMasks",
    "CalculateCompetitionProxy",
    "CalculateEer",
    "CalculateHeadMetrics",
    "CalculateRocAuc",
    "EvaluateSingletonEquivalence",
    "HeadNames",
    "HeadWeights",
)
