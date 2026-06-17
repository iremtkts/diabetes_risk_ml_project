from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

from src.config.path import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from src.data_access.data_loader import DataLoader
from src.evaluation.threshold_analysis import analyze_threshold
from src.inference.artifact_loader import InferenceArtifactLoader
from src.pipelines.training_pipeline import TrainingPipeline
from src.preprocessing.column_config import MODEL_FEATURES, TARGET_COLUMN
from src.preprocessing.splitter import DataSplitter
from src.retraining.promotion_gate import PromotionGate
from src.retraining.retraining_report_writer import (
    ModelSnapshot,
    RetrainingReport,
    RetrainingReportWriter,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 42

CANDIDATE_MODEL_NAME = "xgboost"
CANDIDATE_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}


def main() -> None:
    logger.info("Starting controlled retraining pipeline")

    candidate_model_dir = MODELS_DIR / "candidates" / "latest"
    candidate_report_dir = REPORTS_DIR / "retraining" / "candidate"

    candidate_result = TrainingPipeline(
        data_path=RAW_DATA_DIR / "diabetes.csv",
        model_name=CANDIDATE_MODEL_NAME,
        model_params=CANDIDATE_MODEL_PARAMS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        metrics_output_path=candidate_report_dir / "baseline_metrics.json",
        evaluation_report_output_path=(
            candidate_report_dir / "evaluation_report.json"
        ),
        threshold_report_output_path=(
            candidate_report_dir / "threshold_analysis_report.json"
        ),
        selected_threshold_output_path=(
            candidate_report_dir / "selected_threshold.json"
        ),
        model_output_path=candidate_model_dir / "model.joblib",
        preprocessing_pipeline_output_path=(
            candidate_model_dir / "preprocessing_pipeline.joblib"
        ),
        model_metadata_output_path=candidate_model_dir / "model_metadata.json",
        enable_mlflow_tracking=False,
    ).run()

    production_snapshot = _evaluate_production_model_on_holdout()
    candidate_snapshot = _build_candidate_snapshot(candidate_result)

    promotion_gate_result = PromotionGate().evaluate(
        candidate_metrics=candidate_snapshot.metrics,
        production_metrics=production_snapshot.metrics,
        candidate_confusion_matrix=candidate_snapshot.confusion_matrix,
        production_confusion_matrix=production_snapshot.confusion_matrix,
    )

    retraining_report = RetrainingReport(
        retraining_status="completed",
        candidate_model=candidate_snapshot,
        production_model=production_snapshot,
        promotion_gate=promotion_gate_result,
    )

    output_path = RetrainingReportWriter().write(
        retraining_report=retraining_report,
        output_path=REPORTS_DIR / "retraining" / "retraining_report.json",
    )

    logger.info(
        (
            "Controlled retraining completed | decision=%s | "
            "report=%s | production_artifacts_unchanged=true"
        ),
        promotion_gate_result.promotion_decision,
        output_path,
    )


def _build_candidate_snapshot(candidate_result: Any) -> ModelSnapshot:
    selected_threshold_result = candidate_result.selected_threshold_result

    return ModelSnapshot(
        model_name=CANDIDATE_MODEL_NAME,
        metrics=cast(
            dict[str, float],
            asdict(selected_threshold_result.metrics),
        ),
        confusion_matrix=_to_report_confusion_matrix(
            confusion_matrix_result=selected_threshold_result.confusion_matrix_result
        ),
    )


def _evaluate_production_model_on_holdout() -> ModelSnapshot:
    metadata = InferenceArtifactLoader().load_metadata()
    production_model = InferenceArtifactLoader().load_model(metadata)
    production_preprocessing_pipeline = (
        InferenceArtifactLoader().load_preprocessing_pipeline(metadata)
    )

    train_test_split = _load_holdout_split()
    input_features = cast(
        list[str],
        metadata.get("input_features", MODEL_FEATURES),
    )
    selected_threshold = float(metadata["selected_threshold"])

    X_test_processed = production_preprocessing_pipeline.transform(
        train_test_split.X_test[input_features]
    )
    prediction_probabilities = production_model.predict_proba(
        X_test_processed
    )[:, 1]

    threshold_result = analyze_threshold(
        y_true=train_test_split.y_test,
        prediction_probabilities=prediction_probabilities,
        threshold=selected_threshold,
    )

    return ModelSnapshot(
        model_name=str(metadata.get("model_name", "unknown")),
        metrics=cast(dict[str, float], asdict(threshold_result.metrics)),
        confusion_matrix=_to_report_confusion_matrix(
            confusion_matrix_result=threshold_result.confusion_matrix_result
        ),
    )


def _load_holdout_split() -> Any:
    dataframe = DataLoader().load_data(RAW_DATA_DIR / "diabetes.csv")
    dataset = DataSplitter().split_features_target(
        dataframe=dataframe,
        target_column=TARGET_COLUMN,
    )

    return DataSplitter().split_train_test(
        dataset=dataset,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def _to_report_confusion_matrix(confusion_matrix_result: Any) -> dict[str, int]:
    return {
        "tn": int(confusion_matrix_result.true_negative),
        "fp": int(confusion_matrix_result.false_positive),
        "fn": int(confusion_matrix_result.false_negative),
        "tp": int(confusion_matrix_result.true_positive),
    }


if __name__ == "__main__":
    main()
