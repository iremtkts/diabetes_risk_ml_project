from sklearn.linear_model import LogisticRegression

from src.evaluation.evaluator import (
    EvaluationResult,
    ModelEvaluator,
)


def test_model_evaluator_returns_evaluation_result():
    X_train = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    y_train = [0, 0, 1, 1]

    X_test = [
        [1.5, 1.8],
        [3.5, 3.8],
    ]
    y_test = [0, 1]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()

    result = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    assert isinstance(result, EvaluationResult)


def test_model_evaluator_returns_metrics():
    X_train = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    y_train = [0, 0, 1, 1]

    X_test = [
        [1.5, 1.8],
        [3.5, 3.8],
    ]
    y_test = [0, 1]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()

    result = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    assert result.metrics is not None
    assert result.metrics.accuracy >= 0.0
    assert result.metrics.accuracy <= 1.0
    assert result.metrics.f1 >= 0.0
    assert result.metrics.f1 <= 1.0
    assert result.metrics.roc_auc >= 0.0
    assert result.metrics.roc_auc <= 1.0


def test_model_evaluator_returns_predictions():
    X_train = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    y_train = [0, 0, 1, 1]

    X_test = [
        [1.5, 1.8],
        [3.5, 3.8],
    ]
    y_test = [0, 1]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()

    result = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    assert len(result.predictions) == len(y_test)
    assert len(result.prediction_probabilities) == len(y_test)
    
    
def test_model_evaluator_returns_confusion_matrix_result():
    X_train = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    y_train = [0, 0, 1, 1]

    X_test = [
        [1.5, 1.8],
        [3.5, 3.8],
    ]
    y_test = [0, 1]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()

    result = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    assert result.confusion_matrix_result is not None
    assert result.confusion_matrix_result.true_negative >= 0
    assert result.confusion_matrix_result.false_positive >= 0
    assert result.confusion_matrix_result.false_negative >= 0
    assert result.confusion_matrix_result.true_positive >= 0
    
    
    
    
def test_model_evaluator_confusion_matrix_counts_match_test_size():
    X_train = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    y_train = [0, 0, 1, 1]

    X_test = [
        [1.5, 1.8],
        [3.5, 3.8],
    ]
    y_test = [0, 1]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()

    result = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    confusion_matrix_result = result.confusion_matrix_result

    total_predictions = (
        confusion_matrix_result.true_negative
        + confusion_matrix_result.false_positive
        + confusion_matrix_result.false_negative
        + confusion_matrix_result.true_positive
    )

    assert total_predictions == len(y_test)