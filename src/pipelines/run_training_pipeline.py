from src.config.path import METRICS_DIR, RAW_DATA_DIR
from src.pipelines.training_pipeline import TrainingPipeline


def main() -> None:
    pipeline = TrainingPipeline(
        data_path=RAW_DATA_DIR / "diabetes.csv",
        model_name="logistic_regression",
        model_params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
        metrics_output_path=METRICS_DIR / "baseline_metrics.json",
    )

    result = pipeline.run()

    print("Training completed successfully.")
    print(result.evaluation_result.metrics)


if __name__ == "__main__":
    main()