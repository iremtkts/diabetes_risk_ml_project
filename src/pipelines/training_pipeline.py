from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from src.config.path import RAW_DATA_DIR
from src.data_access.data_loader import DataLoader
from src.evaluation.evaluator import EvaluationResult, ModelEvaluator
from src.models.model_factory import ModelFactory
from src.preprocessing.column_config import TARGET_COLUMN
from src.preprocessing.preprocessing_pipeline import build_preprocessing_pipeline
from src.preprocessing.splitter import DataSplitter
from src.training.trainer import ModelTrainer
from src.utils.logger import get_logger

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
    ) -> None:
        self.data_path = data_path or RAW_DATA_DIR / "diabetes.csv"
        self.model_name = model_name
        self.model_params = model_params or {}
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.data_loader = DataLoader()
        self.data_splitter = DataSplitter()
        self.model_factory = ModelFactory()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()

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