from __future__ import annotations

from src.config.path import REPORTS_DIR
from src.monitoring.monitoring_report_writer import MonitoringReportWriter
from src.monitoring.monitoring_summary import build_monitoring_summary
from src.monitoring.prediction_log_reader import PredictionLogReader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting prediction monitoring summary pipeline")

    prediction_logs = PredictionLogReader().read()
    monitoring_summary = build_monitoring_summary(
        prediction_logs=prediction_logs
    )

    output_path = MonitoringReportWriter().write(
        monitoring_summary=monitoring_summary,
        output_path=(
            REPORTS_DIR
            / "monitoring"
            / "prediction_monitoring_summary.json"
        ),
    )

    logger.info(
        "Prediction monitoring summary pipeline completed | output=%s",
        output_path,
    )


if __name__ == "__main__":
    main()
