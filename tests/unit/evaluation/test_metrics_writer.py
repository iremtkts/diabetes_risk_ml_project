import json

from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.metrics_writer import MetricsWriter


def test_metrics_writer_creates_metrics_file(tmp_path):
    metrics = ClassificationMetrics(
        accuracy=0.75,
        precision=1.0,
        recall=0.5,
        f1=0.66,
        roc_auc=0.9,
    )

    output_path = tmp_path / "metrics" / "baseline_metrics.json"

    writer = MetricsWriter()
    writer.write(metrics=metrics, output_path=output_path)

    assert output_path.exists()


def test_metrics_writer_writes_expected_content(tmp_path):
    metrics = ClassificationMetrics(
        accuracy=0.75,
        precision=1.0,
        recall=0.5,
        f1=0.66,
        roc_auc=0.9,
    )

    output_path = tmp_path / "metrics" / "baseline_metrics.json"

    writer = MetricsWriter()
    writer.write(metrics=metrics, output_path=output_path)

    with output_path.open("r", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert saved_metrics["accuracy"] == 0.75
    assert saved_metrics["precision"] == 1.0
    assert saved_metrics["recall"] == 0.5
    assert saved_metrics["f1"] == 0.66
    assert saved_metrics["roc_auc"] == 0.9