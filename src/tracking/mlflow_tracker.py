from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from src.evaluation.threshold_selector import SelectedThresholdResult
from src.evaluation.evaluator import EvaluationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "diabetes-risk-prediction",
    ) -> None:
        self.experiment_name = experiment_name

    def log_training_run(
        self,
        model_name: str,
        model_params: dict[str, Any],
        test_size: float,
        random_state: int,
        evaluation_result: EvaluationResult,
        selected_threshold_result: SelectedThresholdResult,
        artifact_paths: list[Path],
    ) -> None:
        logger.info("Logging training run to MLflow")

        mlflow.set_experiment(self.experiment_name)

        run_name = f"{model_name}_training_run"

        with mlflow.start_run(run_name=run_name):
            self._log_params(
                model_name=model_name,
                model_params=model_params,
                test_size=test_size,
                random_state=random_state,
                selected_threshold=selected_threshold_result.selected_threshold,
                threshold_selection_strategy=(
                    selected_threshold_result.selection_strategy
                ),
            )

            self._log_default_threshold_metrics(
                evaluation_result=evaluation_result
            )

            self._log_selected_threshold_metrics(
                selected_threshold_result=selected_threshold_result
            )

            self._log_artifacts(artifact_paths=artifact_paths)

        logger.info("Training run logged to MLflow successfully")

    @staticmethod
    def _log_params(
        model_name: str,
        model_params: dict[str, Any],
        test_size: float,
        random_state: int,
        selected_threshold: float,
        threshold_selection_strategy: str,
    ) -> None:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("selected_threshold", selected_threshold)
        mlflow.log_param(
            "threshold_selection_strategy",
            threshold_selection_strategy,
        )

        for param_name, param_value in model_params.items():
            mlflow.log_param(f"model_param_{param_name}", param_value)

    @staticmethod
    def _log_default_threshold_metrics(
        evaluation_result: EvaluationResult,
    ) -> None:
        metrics = evaluation_result.metrics
        confusion_matrix = evaluation_result.confusion_matrix_result

        mlflow.log_metric("default_accuracy", metrics.accuracy)
        mlflow.log_metric("default_precision", metrics.precision)
        mlflow.log_metric("default_recall", metrics.recall)
        mlflow.log_metric("default_f1", metrics.f1)
        mlflow.log_metric("default_roc_auc", metrics.roc_auc)

        mlflow.log_metric(
            "default_true_negative",
            confusion_matrix.true_negative,
        )
        mlflow.log_metric(
            "default_false_positive",
            confusion_matrix.false_positive,
        )
        mlflow.log_metric(
            "default_false_negative",
            confusion_matrix.false_negative,
        )
        mlflow.log_metric(
            "default_true_positive",
            confusion_matrix.true_positive,
        )

    @staticmethod
    def _log_selected_threshold_metrics(
        selected_threshold_result: SelectedThresholdResult,
    ) -> None:
        metrics = selected_threshold_result.metrics
        confusion_matrix = selected_threshold_result.confusion_matrix_result

        mlflow.log_metric("selected_accuracy", metrics.accuracy)
        mlflow.log_metric("selected_precision", metrics.precision)
        mlflow.log_metric("selected_recall", metrics.recall)
        mlflow.log_metric("selected_f1", metrics.f1)
        mlflow.log_metric("selected_roc_auc", metrics.roc_auc)

        mlflow.log_metric(
            "selected_true_negative",
            confusion_matrix.true_negative,
        )
        mlflow.log_metric(
            "selected_false_positive",
            confusion_matrix.false_positive,
        )
        mlflow.log_metric(
            "selected_false_negative",
            confusion_matrix.false_negative,
        )
        mlflow.log_metric(
            "selected_true_positive",
            confusion_matrix.true_positive,
        )

    @staticmethod
    def _log_artifacts(
        artifact_paths: list[Path],
    ) -> None:
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
            else:
                logger.warning(
                    "Skipping missing MLflow artifact: %s",
                    artifact_path,
                )