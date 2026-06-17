from __future__ import annotations

from src.config.path import RAW_DATA_DIR, REPORTS_DIR
from src.data_access.data_loader import DataLoader
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.drift_report_writer import DriftReportWriter
from src.monitoring.prediction_log_reader import PredictionLogReader
from src.monitoring.prediction_logger import DEFAULT_PREDICTION_LOG_PATH
from src.preprocessing.column_config import MODEL_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting drift detection pipeline")

    reference_data_path = RAW_DATA_DIR / "diabetes.csv"
    current_data_path = DEFAULT_PREDICTION_LOG_PATH

    reference_data = DataLoader().load_data(reference_data_path)
    current_logs = PredictionLogReader(log_path=current_data_path).read()

    drift_report = DriftDetector(
        expected_features=MODEL_FEATURES,
        threshold=0.2,
    ).detect(
        reference_data=reference_data,
        current_logs=current_logs,
        reference_data_path=reference_data_path,
        current_data_path=current_data_path,
    )

    output_path = DriftReportWriter().write(
        drift_report=drift_report,
        output_path=REPORTS_DIR / "drift" / "drift_report.json",
    )

    logger.info(
        "Drift detection pipeline completed | output=%s",
        output_path,
    )


if __name__ == "__main__":
    main()
