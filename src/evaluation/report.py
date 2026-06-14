from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.metrics import confusion_matrix


@dataclass
class ConfusionMatrixResult:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def calculate_confusion_matrix_result(
    y_true: Any,
    y_pred: Any,
) -> ConfusionMatrixResult:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return ConfusionMatrixResult(
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )
