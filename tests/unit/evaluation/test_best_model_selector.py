import pytest

from src.evaluation.best_model_selector import select_best_model


def test_select_best_model_prefers_lowest_false_negative() -> None:
    comparison_results = [
        {
            "model_name": "logistic_regression",
            "selected_false_negative": 11,
            "selected_recall": 0.796,
            "selected_f1": 0.683,
            "selected_precision": 0.597,
            "selected_roc_auc": 0.813,
        },
        {
            "model_name": "xgboost",
            "selected_false_negative": 9,
            "selected_recall": 0.833,
            "selected_f1": 0.72,
            "selected_precision": 0.634,
            "selected_roc_auc": 0.83,
        },
    ]

    result = select_best_model(comparison_results=comparison_results)

    assert result.best_model_name == "xgboost"
    assert result.metrics["selected_false_negative"] == 9


def test_select_best_model_uses_recall_as_tiebreaker() -> None:
    comparison_results = [
        {
            "model_name": "model_a",
            "selected_false_negative": 9,
            "selected_recall": 0.80,
            "selected_f1": 0.75,
            "selected_precision": 0.70,
            "selected_roc_auc": 0.81,
        },
        {
            "model_name": "model_b",
            "selected_false_negative": 9,
            "selected_recall": 0.85,
            "selected_f1": 0.72,
            "selected_precision": 0.68,
            "selected_roc_auc": 0.80,
        },
    ]

    result = select_best_model(comparison_results=comparison_results)

    assert result.best_model_name == "model_b"


def test_select_best_model_raises_error_when_results_are_empty() -> None:
    with pytest.raises(ValueError, match="comparison_results cannot be empty"):
        select_best_model(comparison_results=[])