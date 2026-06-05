import json

import pytest

from src.pipelines.training_pipeline import (
    TrainingPipeline,
    TrainingPipelineResult,
)


@pytest.fixture
def training_data_path(tmp_path, large_valid_dataframe):
    data_path = tmp_path / "diabetes.csv"
    large_valid_dataframe.to_csv(data_path, index=False)

    return data_path


@pytest.fixture
def metrics_output_path(tmp_path):
    return tmp_path / "metrics" / "baseline_metrics.json"


@pytest.fixture
def training_pipeline(training_data_path, metrics_output_path):
    return TrainingPipeline(
        data_path=training_data_path,
        model_name="logistic_regression",
        model_params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
        metrics_output_path=metrics_output_path,
    )


def test_training_pipeline_runs_successfully(training_pipeline):
    result = training_pipeline.run()

    assert isinstance(result, TrainingPipelineResult)


def test_training_pipeline_returns_fitted_model(training_pipeline):
    result = training_pipeline.run()

    predictions = result.model.predict(result.X_test_processed)

    assert len(predictions) == len(result.y_test)


def test_training_pipeline_returns_preprocessing_pipeline(training_pipeline):
    result = training_pipeline.run()

    assert result.preprocessing_pipeline is not None


def test_training_pipeline_returns_train_and_test_data(training_pipeline):
    result = training_pipeline.run()

    assert result.X_train_processed is not None
    assert result.X_test_processed is not None
    assert len(result.y_train) > 0
    assert len(result.y_test) > 0


def test_training_pipeline_returns_evaluation_result(training_pipeline):
    result = training_pipeline.run()

    assert result.evaluation_result is not None
    assert result.evaluation_result.metrics is not None
    assert result.evaluation_result.metrics.accuracy >= 0.0
    assert result.evaluation_result.metrics.accuracy <= 1.0


def test_training_pipeline_writes_metrics_artifact(
    training_pipeline,
    metrics_output_path,
):
    training_pipeline.run()

    assert metrics_output_path.exists()

    with metrics_output_path.open("r", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert "accuracy" in saved_metrics
    assert "precision" in saved_metrics
    assert "recall" in saved_metrics
    assert "f1" in saved_metrics
    assert "roc_auc" in saved_metrics