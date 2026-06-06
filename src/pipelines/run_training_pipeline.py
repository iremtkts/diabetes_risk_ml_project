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
    print("X_train_processed shape:", result.X_train_processed.shape)
    print("X_test_processed shape:", result.X_test_processed.shape)

    feature_names = (
        result.preprocessing_pipeline
        .named_steps["column_transformer"]
        .get_feature_names_out()
    )

    coefficients = result.model.coef_[0]

    for feature_name, coefficient in zip(feature_names, coefficients):
        print(feature_name, coefficient)


if __name__ == "__main__":
    main()