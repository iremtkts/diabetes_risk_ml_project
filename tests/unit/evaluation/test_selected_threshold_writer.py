import json

from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.report import ConfusionMatrixResult
from src.evaluation.selected_threshold_writer import SelectedThresholdWriter
from src.evaluation.threshold_selector import SelectedThresholdResult


def test_selected_threshold_writer_writes_json_artifact(tmp_path) -> None:
    output_path = tmp_path / "selected_threshold.json"

    selected_threshold_result = SelectedThresholdResult(
        selected_threshold=0.3,
        selection_strategy="maximize_f1",
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
    )

    writer = SelectedThresholdWriter()

    writer.write(
        selected_threshold_result=selected_threshold_result,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload == {
        "selected_threshold": 0.3,
        "selection_strategy": "maximize_f1",
        "metrics": {
            "accuracy": 0.74,
            "precision": 0.59,
            "recall": 0.79,
            "f1": 0.68,
            "roc_auc": 0.81,
        },
        "confusion_matrix": {
            "true_negative": 71,
            "false_positive": 29,
            "false_negative": 11,
            "true_positive": 43,
        },
    }