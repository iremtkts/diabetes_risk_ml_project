from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.report import ConfusionMatrixResult
from src.evaluation.threshold_analysis import ThresholdAnalysisResult
from src.evaluation.threshold_selector import select_threshold

import pytest


def test_select_threshold_maximize_f1_selects_highest_f1() -> None:
    threshold_analysis_results = [
        ThresholdAnalysisResult(
            threshold=0.3,
            metrics=ClassificationMetrics(
                accuracy=0.74,
                precision=0.59,
                recall=0.79,
                f1=0.68,
                roc_auc=0.81,
            ),
            confusion_matrix_result=ConfusionMatrixResult(
                true_negative=71,
                false_positive=29,
                false_negative=11,
                true_positive=43,
            ),
        ),
        ThresholdAnalysisResult(
            threshold=0.4,
            metrics=ClassificationMetrics(
                accuracy=0.73,
                precision=0.61,
                recall=0.64,
                f1=0.63,
                roc_auc=0.81,
            ),
            confusion_matrix_result=ConfusionMatrixResult(
                true_negative=78,
                false_positive=22,
                false_negative=19,
                true_positive=35,
            ),
        ),
        ThresholdAnalysisResult(
            threshold=0.5,
            metrics=ClassificationMetrics(
                accuracy=0.70,
                precision=0.60,
                recall=0.50,
                f1=0.54,
                roc_auc=0.81,
            ),
            confusion_matrix_result=ConfusionMatrixResult(
                true_negative=82,
                false_positive=18,
                false_negative=27,
                true_positive=27,
            ),
        ),
    ]

    selected_threshold_result = select_threshold(
        threshold_analysis_results=threshold_analysis_results,
        strategy="maximize_f1",
    )

    assert selected_threshold_result.selected_threshold == 0.3
    assert selected_threshold_result.selection_strategy == "maximize_f1"
    assert selected_threshold_result.metrics.f1 == 0.68
    assert selected_threshold_result.confusion_matrix_result.false_negative == 11
    
def test_select_threshold_raises_error_when_results_are_empty() -> None:
    with pytest.raises(ValueError, match="threshold_analysis_results cannot be empty"):
        select_threshold(threshold_analysis_results=[])