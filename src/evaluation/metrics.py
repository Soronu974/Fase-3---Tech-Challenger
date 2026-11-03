"""Métricas de classificação (alvo binário, classe 1 = alfabetização >= limiar)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    out = {
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f1_risco": f1_score(1 - y_true, 1 - pred, zero_division=0),  # classe 0 = risco
        "bal_acc": balanced_accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }
    return {k: float(v) for k, v in out.items()}


def confusion(y_true, proba, threshold: float = 0.5) -> np.ndarray:
    return confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
