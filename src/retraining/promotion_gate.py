from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

PromotionDecision = Literal["approved", "rejected"]


@dataclass(frozen=True)
class PromotionGateResult:
    promotion_decision: PromotionDecision
    checks: dict[str, bool]
    reason: str


class PromotionGate:
    def __init__(self, roc_auc_tolerance: float = 0.02) -> None:
        self.roc_auc_tolerance = roc_auc_tolerance

    def evaluate(
        self,
        candidate_metrics: Mapping[str, float],
        production_metrics: Mapping[str, float],
        candidate_confusion_matrix: Mapping[str, int],
        production_confusion_matrix: Mapping[str, int],
    ) -> PromotionGateResult:
        checks = {
            "recall_not_worse": (
                candidate_metrics["recall"] >= production_metrics["recall"]
            ),
            "f1_not_worse": (
                candidate_metrics["f1"] >= production_metrics["f1"]
            ),
            "roc_auc_within_tolerance": (
                candidate_metrics["roc_auc"]
                >= production_metrics["roc_auc"] - self.roc_auc_tolerance
            ),
            "false_negatives_not_worse": (
                candidate_confusion_matrix["fn"]
                <= production_confusion_matrix["fn"]
            ),
        }

        if all(checks.values()):
            return PromotionGateResult(
                promotion_decision="approved",
                checks=checks,
                reason="Candidate model passed all promotion gate checks.",
            )

        return PromotionGateResult(
            promotion_decision="rejected",
            checks=checks,
            reason="Candidate model failed one or more promotion gate checks.",
        )
