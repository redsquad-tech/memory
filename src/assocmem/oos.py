from __future__ import annotations

from dataclasses import dataclass

from .memory import ReadResult


@dataclass(frozen=True)
class OOSScores:
    """Scores are oriented so that larger means more likely out-of-scope."""

    min_energy: float
    background_responsibility: float
    predictive_uncertainty: float


def raw_oos_scores(read: ReadResult) -> OOSScores:
    min_energy = float("inf") if not read.energies.numel() else float(read.energies.min())
    return OOSScores(
        min_energy=min_energy,
        background_responsibility=float(read.background_responsibility),
        predictive_uncertainty=1.0 - float(read.probabilities.max()),
    )
