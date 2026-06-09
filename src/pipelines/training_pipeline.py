from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from src.config.path import METRICS_DIR, MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from src.data_access.data_loader import DataLoader
from src.evaluation.evaluator import EvaluationResult, ModelEvaluator
from src.evaluation.metrics_writer import MetricsWriter
from src.evaluation.report_writer import EvaluationReportWriter
from src.evaluation.threshold_analysis import analyze_thresholds
from src.evaluation.threshold_report_writer import ThresholdAnalysisReportWriter
from src.models.model_factory import ModelFactory
from src.preprocessing.column_config import TARGET_COLUMN
from src.preprocessing.preprocessing_pipeline import build_preprocessing_pipeline
from src.preprocessing.splitter import DataSplitter
from src.training.trainer import ModelTrainer
from src.evaluation.selected_threshold_writer import SelectedThresholdWriter
from src.evaluation.threshold_selector import select_threshold
from src.utils.logger import get_logger
from src.training.model_artifact_writer import ModelArtifactWriter
from src.tracking.mlflow_tracker import MLflowTracker

logger = get_logger(__name__)


@dataclass
class TrainingPipelineResult:
    model: Any
    preprocessing_pipeline: Pipeline
    X_train_processed: Any
    X_test_processed: Any
    y_train: Any
    y_test: Any
    evaluation_result: EvaluationResult


class TrainingPipeline:
    def __init__(
        self,
        data_path: Path | None = None,
        model_name: str = "logistic_regression",
        model_params: dict[str, Any] | None = None,
        target_column: str = TARGET_COLUMN,
        test_size: float = 0.2,
        random_state: int = 42,
        metrics_output_path: Path | None = None,
        evaluation_report_output_path: Path | None = None,
        thresholds: list[float] | None = None,
        threshold_report_output_path: Path | None = None,
        selected_threshold_output_path: Path | None = None,
        model_output_path: Path | None = None,
        preprocessing_pipeline_output_path: Path | None = None,
        model_metadata_output_path: Path | None = None,
        enable_mlflow_tracking: bool = True,
    ) -> None:
        self.data_path = data_path or RAW_DATA_DIR / "diabetes.csv"
        self.model_name = model_name
        self.model_params = model_params or {}
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.enable_mlflow_tracking = enable_mlflow_tracking

        self.metrics_output_path = (
            metrics_output_path or METRICS_DIR / "baseline_metrics.json"
        )

        self.evaluation_report_output_path = (
            evaluation_report_output_path
            or REPORTS_DIR / "evaluation_report.json"
        )

        self.thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]

        self.threshold_report_output_path = (
            threshold_report_output_path
            or REPORTS_DIR / "threshold_analysis_report.json"
        )
        self.selected_threshold_output_path = (
          selected_threshold_output_path
          or REPORTS_DIR / "selected_threshold.json"
        )
        
        self.model_output_path = model_output_path or MODELS_DIR / "model.joblib"

        self.preprocessing_pipeline_output_path = (
           preprocessing_pipeline_output_path
             or MODELS_DIR / "preprocessing_pipeline.joblib"
        )

        self.model_metadata_output_path = (
         model_metadata_output_path
           or MODELS_DIR / "model_metadata.json"
        )

        self.data_loader = DataLoader()
        self.data_splitter = DataSplitter()
        self.model_factory = ModelFactory()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.metrics_writer = MetricsWriter()
        self.evaluation_report_writer = EvaluationReportWriter()
        self.threshold_analysis_report_writer = ThresholdAnalysisReportWriter()
        self.selected_threshold_writer = SelectedThresholdWriter()
        self.model_artifact_writer = ModelArtifactWriter()
        self.mlflow_tracker = MLflowTracker()
        
        
        
    @staticmethod
    def _get_processed_feature_names(
        preprocessing_pipeline: Pipeline,
    ) -> list[str]:
        column_transformer = preprocessing_pipeline.named_steps["column_transformer"]

        return column_transformer.get_feature_names_out().tolist()

    def run(self) -> TrainingPipelineResult:
        logger.info("Starting training pipeline")
        logger.info("Loading data from: %s", self.data_path)

        dataframe = self.data_loader.load_data(self.data_path)

        dataset = self.data_splitter.split_features_target(
            dataframe=dataframe,
            target_column=self.target_column,
        )

        train_test_split = self.data_splitter.split_train_test(
            dataset=dataset,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        logger.info("Building preprocessing pipeline")
        preprocessing_pipeline = build_preprocessing_pipeline()

        logger.info("Fitting preprocessing pipeline on training data")
        X_train_processed = preprocessing_pipeline.fit_transform(
            train_test_split.X_train
        )

        logger.info("Transforming test data")
        X_test_processed = preprocessing_pipeline.transform(
            train_test_split.X_test
        )

        logger.info("Creating model: %s", self.model_name)
        model = self.model_factory.create_model(
            model_name=self.model_name,
            **self.model_params,
        )

        logger.info("Training model")
        training_result = self.model_trainer.train(
            model=model,
            X_train=X_train_processed,
            y_train=train_test_split.y_train,
        )

        logger.info("Evaluating model")
        evaluation_result = self.model_evaluator.evaluate(
            model=training_result.model,
            X_test=X_test_processed,
            y_test=train_test_split.y_test,
        )

        logger.info("Writing evaluation metrics")
        self.metrics_writer.write(
            metrics=evaluation_result.metrics,
            output_path=self.metrics_output_path,
        )

        logger.info("Writing evaluation report")
        self.evaluation_report_writer.write(
            evaluation_result=evaluation_result,
            output_path=self.evaluation_report_output_path,
        )

        logger.info("Running threshold analysis")
        threshold_analysis_results = analyze_thresholds(
            y_true=train_test_split.y_test,
            prediction_probabilities=evaluation_result.prediction_probabilities,
            thresholds=self.thresholds,
        )

        logger.info("Writing threshold analysis report")
        self.threshold_analysis_report_writer.write(
            threshold_analysis_results=threshold_analysis_results,
            output_path=self.threshold_report_output_path,
        )
        
        
        logger.info("Selecting best threshold")
        selected_threshold_result = select_threshold(
          threshold_analysis_results=threshold_analysis_results,
          strategy="maximize_f1",
         )

        logger.info(
          "Selected threshold=%s using strategy=%s",
          selected_threshold_result.selected_threshold,
          selected_threshold_result.selection_strategy,
        )

        logger.info("Writing selected threshold artifact")
        self.selected_threshold_writer.write(
          selected_threshold_result=selected_threshold_result,
        output_path=self.selected_threshold_output_path,
        )
        
        
        logger.info("Saving model artifacts")
        model_path = self.model_artifact_writer.save_model(
        model=training_result.model,
        output_path=self.model_output_path,
        )

        preprocessing_pipeline_path = (
        self.model_artifact_writer.save_preprocessing_pipeline(
        preprocessing_pipeline=preprocessing_pipeline,
        output_path=self.preprocessing_pipeline_output_path,
           )
        )

        processed_features = self._get_processed_feature_names(
        preprocessing_pipeline=preprocessing_pipeline
        )

        model_metadata_path = self.model_artifact_writer.save_metadata(
        model_name=self.model_name,
        model_path=model_path,
        preprocessing_pipeline_path=preprocessing_pipeline_path,
        selected_threshold=selected_threshold_result.selected_threshold,
        processed_features=processed_features,
        output_path=self.model_metadata_output_path,
        )

        
        
        if self.enable_mlflow_tracking:
          logger.info("Logging training run to MLflow")
          self.mlflow_tracker.log_training_run(
          model_name=self.model_name,
          model_params=self.model_params,
          test_size=self.test_size,
          random_state=self.random_state,
          evaluation_result=evaluation_result,
          selected_threshold_result=selected_threshold_result,
          artifact_paths=[
            self.metrics_output_path,
            self.evaluation_report_output_path,
            self.threshold_report_output_path,
            self.selected_threshold_output_path,
            model_path,
            preprocessing_pipeline_path,
            model_metadata_path,
           ],
        )
          
        logger.info("Training pipeline completed successfully")

        return TrainingPipelineResult(
            model=training_result.model,
            preprocessing_pipeline=preprocessing_pipeline,
            X_train_processed=X_train_processed,
            X_test_processed=X_test_processed,
            y_train=train_test_split.y_train,
            y_test=train_test_split.y_test,
            evaluation_result=evaluation_result,
        )