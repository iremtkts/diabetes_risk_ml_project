import json

from src.evaluation.evaluator import EvaluationResult
from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.report import ConfusionMatrixResult
from src.evaluation.report_writer import EvaluationReportWriter


def test_evaluation_report_writer_creates_report_file(tmp_path):
    evaluation_result = EvaluationResult(
        metrics=ClassificationMetrics(
            accuracy=0.75,
            precision=0.8,
            recall=0.6,
            f1=0.68,
            roc_auc=0.85,
        ),
        confusion_matrix_result=ConfusionMatrixResult(
            true_negative=50,
            false_positive=10,
            false_negative=8,
            true_positive=32,
        ),
        predictions=[0, 1, 0, 1],
        prediction_probabilities=[0.1, 0.8, 0.3, 0.9],
    )

    output_path = tmp_path / "reports" / "evaluation_report.json"

    writer = EvaluationReportWriter()
    writer.write(
        evaluation_result=evaluation_result,
        output_path=output_path,
    )

    assert output_path.exists()


def test_evaluation_report_writer_writes_expected_content(tmp_path):
    evaluation_result = EvaluationResult(
        metrics=ClassificationMetrics(
            accuracy=0.75,
            precision=0.8,
            recall=0.6,
            f1=0.68,
            roc_auc=0.85,
        ),
        confusion_matrix_result=ConfusionMatrixResult(
            true_negative=50,
            false_positive=10,
            false_negative=8,
            true_positive=32,
        ),
        predictions=[0, 1, 0, 1],
        prediction_probabilities=[0.1, 0.8, 0.3, 0.9],
    )

    output_path = tmp_path / "reports" / "evaluation_report.json"

    writer = EvaluationReportWriter()
    writer.write(
        evaluation_result=evaluation_result,
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert "metrics" in saved_report
    assert "confusion_matrix" in saved_report

    assert saved_report["metrics"]["accuracy"] == 0.75
    assert saved_report["metrics"]["precision"] == 0.8
    assert saved_report["metrics"]["recall"] == 0.6
    assert saved_report["metrics"]["f1"] == 0.68
    assert saved_report["metrics"]["roc_auc"] == 0.85

    assert saved_report["confusion_matrix"]["true_negative"] == 50
    assert saved_report["confusion_matrix"]["false_positive"] == 10
    assert saved_report["confusion_matrix"]["false_negative"] == 8
    assert saved_report["confusion_matrix"]["true_positive"] == 32