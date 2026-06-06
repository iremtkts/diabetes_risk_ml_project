from src.evaluation.threshold_analysis import (
    ThresholdAnalysisResult,
    analyze_threshold,
    analyze_thresholds,
    predictions_from_threshold,
)


def test_predictions_from_threshold():
    prediction_probabilities = [0.1, 0.4, 0.6, 0.9]

    predictions = predictions_from_threshold(
        prediction_probabilities=prediction_probabilities,
        threshold=0.5,
    )

    assert predictions.tolist() == [0, 0, 1, 1]


def test_predictions_from_threshold_includes_equal_threshold():
    prediction_probabilities = [0.3, 0.5, 0.7]

    predictions = predictions_from_threshold(
        prediction_probabilities=prediction_probabilities,
        threshold=0.5,
    )

    assert predictions.tolist() == [0, 1, 1]


def test_analyze_threshold_returns_result():
    y_true = [0, 0, 1, 1]
    prediction_probabilities = [0.1, 0.6, 0.4, 0.9]

    result = analyze_threshold(
        y_true=y_true,
        prediction_probabilities=prediction_probabilities,
        threshold=0.5,
    )

    assert isinstance(result, ThresholdAnalysisResult)


def test_analyze_threshold_calculates_confusion_matrix():
    y_true = [0, 0, 1, 1]
    prediction_probabilities = [0.1, 0.6, 0.4, 0.9]

    result = analyze_threshold(
        y_true=y_true,
        prediction_probabilities=prediction_probabilities,
        threshold=0.5,
    )

    assert result.confusion_matrix_result.true_negative == 1
    assert result.confusion_matrix_result.false_positive == 1
    assert result.confusion_matrix_result.false_negative == 1
    assert result.confusion_matrix_result.true_positive == 1


def test_analyze_thresholds_returns_one_result_per_threshold():
    y_true = [0, 0, 1, 1]
    prediction_probabilities = [0.1, 0.6, 0.4, 0.9]
    thresholds = [0.3, 0.5, 0.7]

    results = analyze_thresholds(
        y_true=y_true,
        prediction_probabilities=prediction_probabilities,
        thresholds=thresholds,
    )

    assert len(results) == len(thresholds)
    assert [result.threshold for result in results] == thresholds