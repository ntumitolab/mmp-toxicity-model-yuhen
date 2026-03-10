"""
Evaluation metrics and Excel export.
"""
import os
from typing import Dict, List, Union

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, Union[int, float]]:
    """
    Compute a standard set of binary-classification metrics.

    Returns
    -------
    dict with keys: TP, FP, FN, TN, F1 Score, MCC, recall, precision,
    specificity, accuracy, aucroc, Balanced Accuracy.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "TP":                int(tp),
        "FP":                int(fp),
        "FN":                int(fn),
        "TN":                int(tn),
        "F1 Score":          f1_score(y_true, y_pred, zero_division=0),
        "MCC":               matthews_corrcoef(y_true, y_pred),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "specificity":       specificity,
        "accuracy":          accuracy_score(y_true, y_pred),
        "aucroc":            roc_auc_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def save_to_excel(
    metrics: Dict[str, Union[int, float]],
    path: str,
) -> None:
    """
    Write a metrics dict to an Excel file at *path*.

    The directory is created if it does not exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([metrics])
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="Metrics", index=False)
