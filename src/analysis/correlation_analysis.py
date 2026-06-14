from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.path import ARTIFACTS_DIR, RAW_DATA_DIR
from src.data_access.data_loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = ARTIFACTS_DIR / "reports"


def calculate_correlation_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.corr(numeric_only=True)


def save_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correlation_matrix.to_csv(output_path)
    logger.info("Correlation matrix saved to: %s", output_path)


def save_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )
    plt.title("Diabetes Dataset Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info("Correlation heatmap saved to: %s", output_path)


def main() -> None:
    data_loader = DataLoader()
    dataframe = data_loader.load_data(RAW_DATA_DIR / "diabetes.csv")

    correlation_matrix = calculate_correlation_matrix(dataframe)

    save_correlation_matrix(
        correlation_matrix=correlation_matrix,
        output_path=REPORTS_DIR / "correlation_matrix.csv",
    )

    save_correlation_heatmap(
        correlation_matrix=correlation_matrix,
        output_path=REPORTS_DIR / "correlation_heatmap.png",
    )


if __name__ == "__main__":
    main()