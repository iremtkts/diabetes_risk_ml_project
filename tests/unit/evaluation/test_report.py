from src.evaluation.report import (
    ConfusionMatrixResult,
    calculate_confusion_matrix_result,
)


def test_calculate_confusion_matrix_result_returns_result():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    result = calculate_confusion_matrix_result(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert isinstance(result, ConfusionMatrixResult)


def test_calculate_confusion_matrix_result_values():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]

    result = calculate_confusion_matrix_result(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert result.true_negative == 2
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.true_positive == 2