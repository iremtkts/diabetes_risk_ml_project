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
def evaluation_report_output_path(tmp_path):
    return tmp_path / "reports" / "evaluation_report.json"


@pytest.fixture
def threshold_report_output_path(tmp_path):
    return tmp_path / "reports" / "threshold_analysis_report.json"


@pytest.fixture
def selected_threshold_output_path(tmp_path):
    return tmp_path / "reports" / "selected_threshold.json"


@pytest.fixture
def model_output_path(tmp_path):
    return tmp_path / "models" / "model.joblib"


@pytest.fixture
def preprocessing_pipeline_output_path(tmp_path):
    return tmp_path / "models" / "preprocessing_pipeline.joblib"


@pytest.fixture
def model_metadata_output_path(tmp_path):
    return tmp_path / "models" / "model_metadata.json"


@pytest.fixture
def training_pipeline(
    training_data_path,
    metrics_output_path,
    evaluation_report_output_path,
    threshold_report_output_path,
    selected_threshold_output_path,
    model_output_path,
    preprocessing_pipeline_output_path,
    model_metadata_output_path,
):
    return TrainingPipeline(
        data_path=training_data_path,
        model_name="logistic_regression",
        model_params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
        metrics_output_path=metrics_output_path,
        evaluation_report_output_path=evaluation_report_output_path,
        threshold_report_output_path=threshold_report_output_path,
        selected_threshold_output_path=selected_threshold_output_path,
        model_output_path=model_output_path,
        preprocessing_pipeline_output_path=preprocessing_pipeline_output_path,
        model_metadata_output_path=model_metadata_output_path,
        enable_mlflow_tracking=False,
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


def test_training_pipeline_writes_evaluation_report_artifact(
    training_pipeline,
    evaluation_report_output_path,
):
    training_pipeline.run()

    assert evaluation_report_output_path.exists()

    with evaluation_report_output_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert "metrics" in saved_report
    assert "confusion_matrix" in saved_report

    assert "accuracy" in saved_report["metrics"]
    assert "precision" in saved_report["metrics"]
    assert "recall" in saved_report["metrics"]
    assert "f1" in saved_report["metrics"]
    assert "roc_auc" in saved_report["metrics"]

    assert "true_negative" in saved_report["confusion_matrix"]
    assert "false_positive" in saved_report["confusion_matrix"]
    assert "false_negative" in saved_report["confusion_matrix"]
    assert "true_positive" in saved_report["confusion_matrix"]


def test_training_pipeline_writes_threshold_analysis_report_artifact(
    training_pipeline,
    threshold_report_output_path,
):
    training_pipeline.run()

    assert threshold_report_output_path.exists()

    with threshold_report_output_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert isinstance(saved_report, list)
    assert len(saved_report) > 0

    first_threshold_result = saved_report[0]

    assert "threshold" in first_threshold_result
    assert "metrics" in first_threshold_result
    assert "confusion_matrix" in first_threshold_result


def test_training_pipeline_writes_selected_threshold_artifact(
    training_pipeline,
    selected_threshold_output_path,
):
    training_pipeline.run()

    assert selected_threshold_output_path.exists()

    with selected_threshold_output_path.open("r", encoding="utf-8") as file:
        saved_selected_threshold = json.load(file)

    assert "selected_threshold" in saved_selected_threshold
    assert "selection_strategy" in saved_selected_threshold
    assert "metrics" in saved_selected_threshold
    assert "confusion_matrix" in saved_selected_threshold


def test_training_pipeline_writes_model_artifacts(
    training_pipeline,
    model_output_path,
    preprocessing_pipeline_output_path,
    model_metadata_output_path,
):
    training_pipeline.run()

    assert model_output_path.exists()
    assert preprocessing_pipeline_output_path.exists()
    assert model_metadata_output_path.exists()

    with model_metadata_output_path.open("r", encoding="utf-8") as file:
        saved_metadata = json.load(file)

    assert saved_metadata["model_name"] == "logistic_regression"
    assert "model_path" in saved_metadata
    assert "preprocessing_pipeline_path" in saved_metadata
    assert "selected_threshold" in saved_metadata
    assert "input_features" in saved_metadata
    assert "processed_features" in saved_metadata