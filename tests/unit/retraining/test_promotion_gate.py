from src.retraining.promotion_gate import PromotionGate


def test_promotion_gate_approves_when_all_checks_pass() -> None:
    result = PromotionGate().evaluate(
        candidate_metrics=_metrics(recall=0.8, f1=0.7, roc_auc=0.79),
        production_metrics=_metrics(recall=0.8, f1=0.7, roc_auc=0.8),
        candidate_confusion_matrix=_confusion_matrix(fn=10),
        production_confusion_matrix=_confusion_matrix(fn=10),
    )

    assert result.promotion_decision == "approved"
    assert all(result.checks.values())


def test_promotion_gate_rejects_when_recall_is_worse() -> None:
    result = PromotionGate().evaluate(
        candidate_metrics=_metrics(recall=0.79),
        production_metrics=_metrics(recall=0.8),
        candidate_confusion_matrix=_confusion_matrix(),
        production_confusion_matrix=_confusion_matrix(),
    )

    assert result.promotion_decision == "rejected"
    assert result.checks["recall_not_worse"] is False


def test_promotion_gate_rejects_when_f1_is_worse() -> None:
    result = PromotionGate().evaluate(
        candidate_metrics=_metrics(f1=0.69),
        production_metrics=_metrics(f1=0.7),
        candidate_confusion_matrix=_confusion_matrix(),
        production_confusion_matrix=_confusion_matrix(),
    )

    assert result.promotion_decision == "rejected"
    assert result.checks["f1_not_worse"] is False


def test_promotion_gate_rejects_when_roc_auc_is_too_low() -> None:
    result = PromotionGate().evaluate(
        candidate_metrics=_metrics(roc_auc=0.77),
        production_metrics=_metrics(roc_auc=0.8),
        candidate_confusion_matrix=_confusion_matrix(),
        production_confusion_matrix=_confusion_matrix(),
    )

    assert result.promotion_decision == "rejected"
    assert result.checks["roc_auc_within_tolerance"] is False


def test_promotion_gate_rejects_when_false_negatives_are_worse() -> None:
    result = PromotionGate().evaluate(
        candidate_metrics=_metrics(),
        production_metrics=_metrics(),
        candidate_confusion_matrix=_confusion_matrix(fn=11),
        production_confusion_matrix=_confusion_matrix(fn=10),
    )

    assert result.promotion_decision == "rejected"
    assert result.checks["false_negatives_not_worse"] is False


def test_promotion_gate_allows_roc_auc_within_tolerance() -> None:
    result = PromotionGate(roc_auc_tolerance=0.02).evaluate(
        candidate_metrics=_metrics(roc_auc=0.78),
        production_metrics=_metrics(roc_auc=0.8),
        candidate_confusion_matrix=_confusion_matrix(),
        production_confusion_matrix=_confusion_matrix(),
    )

    assert result.promotion_decision == "approved"
    assert result.checks["roc_auc_within_tolerance"] is True


def _metrics(
    accuracy: float = 0.75,
    precision: float = 0.7,
    recall: float = 0.8,
    f1: float = 0.7,
    roc_auc: float = 0.8,
) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def _confusion_matrix(
    tn: int = 70,
    fp: int = 30,
    fn: int = 10,
    tp: int = 44,
) -> dict[str, int]:
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
