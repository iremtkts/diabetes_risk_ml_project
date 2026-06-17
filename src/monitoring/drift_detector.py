from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.path import ROOT_DIR
from src.preprocessing.column_config import MODEL_FEATURES


@dataclass(frozen=True)
class SchemaDriftResult:
    missing_features: list[str]
    unexpected_features: list[str]
    schema_drift_detected: bool


@dataclass(frozen=True)
class FeatureDriftResult:
    feature_name: str
    reference_mean: float | None
    current_mean: float | None
    absolute_difference: float | None
    relative_difference: float | None
    drift_detected: bool
    calculation_available: bool
    reason: str | None = None


@dataclass(frozen=True)
class DataDriftResult:
    threshold: float
    data_drift_detected: bool
    features: list[FeatureDriftResult]


@dataclass(frozen=True)
class DriftReport:
    reference_data_path: str
    current_data_path: str
    total_current_records: int
    schema_drift: SchemaDriftResult
    data_drift: DataDriftResult


class DriftDetector:
    def __init__(
        self,
        expected_features: list[str] | None = None,
        threshold: float = 0.2,
    ) -> None:
        self.expected_features = expected_features or MODEL_FEATURES
        self.threshold = threshold

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_logs: list[dict[str, Any]],
        reference_data_path: Path,
        current_data_path: Path,
    ) -> DriftReport:
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")

        current_feature_records = self._extract_current_features(current_logs)
        schema_drift = self._detect_schema_drift(current_feature_records)
        data_drift = self._detect_data_drift(
            reference_data=reference_data,
            current_feature_records=current_feature_records,
        )

        return DriftReport(
            reference_data_path=self._to_root_relative_path(reference_data_path),
            current_data_path=self._to_root_relative_path(current_data_path),
            total_current_records=len(current_feature_records),
            schema_drift=schema_drift,
            data_drift=data_drift,
        )

    def _extract_current_features(
        self,
        current_logs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not current_logs:
            raise ValueError("Current prediction logs cannot be empty")

        current_feature_records: list[dict[str, Any]] = []

        for index, record in enumerate(current_logs, start=1):
            input_features = record.get("input_features")

            if not isinstance(input_features, dict):
                raise ValueError(
                    "Prediction log record is missing a valid "
                    f"input_features object at position {index}"
                )

            current_feature_records.append(input_features)

        return current_feature_records

    def _detect_schema_drift(
        self,
        current_feature_records: list[dict[str, Any]],
    ) -> SchemaDriftResult:
        expected_feature_set = set(self.expected_features)
        current_feature_set = {
            feature_name
            for record in current_feature_records
            for feature_name in record
        }

        missing_features = sorted(expected_feature_set - current_feature_set)
        unexpected_features = sorted(current_feature_set - expected_feature_set)

        return SchemaDriftResult(
            missing_features=missing_features,
            unexpected_features=unexpected_features,
            schema_drift_detected=bool(missing_features or unexpected_features),
        )

    def _detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_feature_records: list[dict[str, Any]],
    ) -> DataDriftResult:
        feature_results = [
            self._calculate_feature_drift(
                feature_name=feature_name,
                reference_data=reference_data,
                current_feature_records=current_feature_records,
            )
            for feature_name in self.expected_features
        ]

        return DataDriftResult(
            threshold=self.threshold,
            data_drift_detected=any(
                result.drift_detected for result in feature_results
            ),
            features=feature_results,
        )

    def _calculate_feature_drift(
        self,
        feature_name: str,
        reference_data: pd.DataFrame,
        current_feature_records: list[dict[str, Any]],
    ) -> FeatureDriftResult:
        if feature_name not in reference_data.columns:
            return self._unavailable_feature_result(
                feature_name=feature_name,
                reason="missing_reference_feature",
            )

        current_values = [
            float(record[feature_name])
            for record in current_feature_records
            if self._is_numeric_feature_value(record.get(feature_name))
        ]

        if not current_values:
            return self._unavailable_feature_result(
                feature_name=feature_name,
                reason="missing_current_feature",
            )

        reference_mean = float(reference_data[feature_name].mean())
        current_mean = float(pd.Series(current_values).mean())
        absolute_difference = abs(current_mean - reference_mean)
        relative_difference = self._calculate_relative_difference(
            reference_mean=reference_mean,
            absolute_difference=absolute_difference,
        )

        return FeatureDriftResult(
            feature_name=feature_name,
            reference_mean=reference_mean,
            current_mean=current_mean,
            absolute_difference=absolute_difference,
            relative_difference=relative_difference,
            drift_detected=relative_difference > self.threshold,
            calculation_available=True,
        )

    def _calculate_relative_difference(
        self,
        reference_mean: float,
        absolute_difference: float,
    ) -> float:
        if reference_mean == 0:
            return 0.0 if absolute_difference == 0 else 1.0

        return absolute_difference / abs(reference_mean)

    def _unavailable_feature_result(
        self,
        feature_name: str,
        reason: str,
    ) -> FeatureDriftResult:
        return FeatureDriftResult(
            feature_name=feature_name,
            reference_mean=None,
            current_mean=None,
            absolute_difference=None,
            relative_difference=None,
            drift_detected=False,
            calculation_available=False,
            reason=reason,
        )

    def _is_numeric_feature_value(self, value: Any) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool)

    def _to_root_relative_path(self, path: Path) -> str:
        return os.path.relpath(path.resolve(), ROOT_DIR).replace(os.sep, "/")
