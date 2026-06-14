from __future__ import annotations

from typing import Any, TypedDict

from src.config.path import MODELS_DIR, REPORTS_DIR
from src.evaluation.best_model_report_writer import BestModelReportWriter
from src.evaluation.best_model_selector import select_best_model
from src.evaluation.model_comparison_report_writer import (
    ModelComparisonReportWriter,
)
from src.pipelines.training_pipeline import TrainingPipeline
from src.training.model_promoter import ModelPromoter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExperiment(TypedDict):
    model_name: str
    model_params: dict[str, Any]


MODEL_EXPERIMENTS: list[ModelExperiment] = [
    {
        "model_name": "logistic_regression",
        "model_params": {
            "max_iter": 1000,
        },
    },
    {
        "model_name": "random_forest",
        "model_params": {
            "n_estimators": 200,
            "max_depth": 5,
            "random_state": 42,
        },
    },
    {
        "model_name": "xgboost",
        "model_params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": 42,
        },
    },
]


def build_comparison_row(
    model_name: str,
    result: Any,
) -> dict[str, Any]:
    default_metrics = result.evaluation_result.metrics
    default_confusion_matrix = result.evaluation_result.confusion_matrix_result

    selected_threshold_result = result.selected_threshold_result
    selected_metrics = selected_threshold_result.metrics
    selected_confusion_matrix = (
        selected_threshold_result.confusion_matrix_result
    )

    return {
        "model_name": model_name,
        "selected_threshold": selected_threshold_result.selected_threshold,
        "default_accuracy": default_metrics.accuracy,
        "default_precision": default_metrics.precision,
        "default_recall": default_metrics.recall,
        "default_f1": default_metrics.f1,
        "default_roc_auc": default_metrics.roc_auc,
        "default_false_negative": default_confusion_matrix.false_negative,
        "default_false_positive": default_confusion_matrix.false_positive,
        "selected_accuracy": selected_metrics.accuracy,
        "selected_precision": selected_metrics.precision,
        "selected_recall": selected_metrics.recall,
        "selected_f1": selected_metrics.f1,
        "selected_roc_auc": selected_metrics.roc_auc,
        "selected_false_negative": selected_confusion_matrix.false_negative,
        "selected_false_positive": selected_confusion_matrix.false_positive,
    }


def main() -> None:
    comparison_results = []

    for experiment in MODEL_EXPERIMENTS:
        model_name = experiment["model_name"]
        model_params = experiment["model_params"]
        model_artifact_dir = MODELS_DIR / model_name
        model_report_dir = REPORTS_DIR / model_name

        logger.info("Starting model comparison run for: %s", model_name)

        pipeline = TrainingPipeline(
        model_name=model_name,
        model_params=model_params,
        test_size=0.2,
        random_state=42,
        metrics_output_path=model_report_dir / "baseline_metrics.json",
        evaluation_report_output_path=model_report_dir / "evaluation_report.json",
        threshold_report_output_path=(
          model_report_dir / "threshold_analysis_report.json"
        ),
        selected_threshold_output_path=model_report_dir / "selected_threshold.json",
        model_output_path=model_artifact_dir / "model.joblib",
        preprocessing_pipeline_output_path=(
          model_artifact_dir / "preprocessing_pipeline.joblib"
        ),
        model_metadata_output_path=model_artifact_dir / "model_metadata.json",
        enable_mlflow_tracking=True,
        )

        result = pipeline.run()

        comparison_row = build_comparison_row(
            model_name=model_name,
            result=result,
        )
        comparison_results.append(comparison_row)

        logger.info(
    (
        "Completed model comparison run | model=%s | "
        "selected_f1=%.4f | selected_recall=%.4f | "
        "selected_fn=%s"
    ),
    model_name,
    comparison_row["selected_f1"],
    comparison_row["selected_recall"],
    comparison_row["selected_false_negative"],
)

    comparison_report_writer = ModelComparisonReportWriter()
    comparison_report_writer.write_json(comparison_results=comparison_results)
    comparison_report_writer.write_csv(comparison_results=comparison_results)

    best_model_selection_result = select_best_model(
    comparison_results=comparison_results
    )

    best_model_report_writer = BestModelReportWriter()
    best_model_report_writer.write(
    best_model_selection_result=best_model_selection_result
    )
    
    best_model_name = best_model_selection_result.best_model_name
    best_model_artifact_dir = MODELS_DIR / best_model_name

    model_promoter = ModelPromoter()
    production_model_dir = model_promoter.promote(
      source_model_dir=best_model_artifact_dir
    )

    logger.info(
    "Best model promoted to production | model=%s | production_dir=%s",
    best_model_name,
    production_model_dir,
)

    logger.info(
    (
        "Best model selected | model=%s | selected_f1=%.4f | "
        "selected_recall=%.4f | selected_fn=%s"
    ),
    best_model_selection_result.best_model_name,
    best_model_selection_result.metrics["selected_f1"],
    best_model_selection_result.metrics["selected_recall"],
    best_model_selection_result.metrics["selected_false_negative"],
)

    logger.info("Model comparison completed successfully")


if __name__ == "__main__":
    main()
