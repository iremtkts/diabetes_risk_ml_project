import json

from src.retraining.promotion_gate import PromotionGateResult
from src.retraining.retraining_report_writer import (
    ModelSnapshot,
    RetrainingReport,
    RetrainingReportWriter,
)


def test_retraining_report_writer_creates_output_directory(tmp_path) -> None:
    output_path = tmp_path / "reports" / "retraining" / "report.json"
    writer = RetrainingReportWriter()

    writer.write(
        retraining_report=_sample_retraining_report(),
        output_path=output_path,
    )

    assert output_path.exists()


def test_retraining_report_writer_writes_expected_json(tmp_path) -> None:
    output_path = tmp_path / "retraining_report.json"
    writer = RetrainingReportWriter()

    writer.write(
        retraining_report=_sample_retraining_report(),
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload == {
        "retraining_status": "completed",
        "candidate_model": {
            "model_name": "xgboost",
            "metrics": {
                "accuracy": 0.74,
                "precision": 0.6,
                "recall": 0.79,
                "f1": 0.68,
                "roc_auc": 0.81,
            },
            "confusion_matrix": {
                "tn": 71,
                "fp": 29,
                "fn": 11,
                "tp": 43,
            },
        },
        "production_model": {
            "model_name": "xgboost",
            "metrics": {
                "accuracy": 0.73,
                "precision": 0.59,
                "recall": 0.78,
                "f1": 0.67,
                "roc_auc": 0.8,
            },
            "confusion_matrix": {
                "tn": 70,
                "fp": 30,
                "fn": 12,
                "tp": 42,
            },
        },
        "promotion_gate": {
            "promotion_decision": "approved",
            "checks": {
                "recall_not_worse": True,
                "f1_not_worse": True,
                "roc_auc_within_tolerance": True,
                "false_negatives_not_worse": True,
            },
            "reason": "Candidate model passed all promotion gate checks.",
        },
    }


def _sample_retraining_report() -> RetrainingReport:
    return RetrainingReport(
        retraining_status="completed",
        candidate_model=ModelSnapshot(
            model_name="xgboost",
            metrics={
                "accuracy": 0.74,
                "precision": 0.6,
                "recall": 0.79,
                "f1": 0.68,
                "roc_auc": 0.81,
            },
            confusion_matrix={
                "tn": 71,
                "fp": 29,
                "fn": 11,
                "tp": 43,
            },
        ),
        production_model=ModelSnapshot(
            model_name="xgboost",
            metrics={
                "accuracy": 0.73,
                "precision": 0.59,
                "recall": 0.78,
                "f1": 0.67,
                "roc_auc": 0.8,
            },
            confusion_matrix={
                "tn": 70,
                "fp": 30,
                "fn": 12,
                "tp": 42,
            },
        ),
        promotion_gate=PromotionGateResult(
            promotion_decision="approved",
            checks={
                "recall_not_worse": True,
                "f1_not_worse": True,
                "roc_auc_within_tolerance": True,
                "false_negatives_not_worse": True,
            },
            reason="Candidate model passed all promotion gate checks.",
        ),
    )
