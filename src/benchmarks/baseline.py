"""Static logistic-regression benchmark on the same window features."""

from __future__ import annotations

import numpy as np


def fit_baseline(features: np.ndarray, labels: np.ndarray) -> object:
    """Fit a balanced logistic-regression baseline and return the estimator."""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("baseline evaluation requires scikit-learn") from exc
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be 2-D and labels must be a matching 1-D array")
    if len(np.unique(labels)) < 2:
        raise ValueError("logistic regression requires both benign and attack labels")
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(features, labels)


def evaluate_baseline(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Fit/evaluate logistic regression and return comparable binary metrics."""

    model = fit_baseline(features, labels)
    return metrics_from_predictions(labels, model.predict(features))


def metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Calculate the common metric contract for any model's predictions."""

    try:
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    except ImportError as exc:
        raise RuntimeError("benchmark metrics require scikit-learn") from exc
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    tn, fp, _, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def compare_metrics(
    baseline: dict[str, float], world_model: dict[str, float]
) -> list[dict[str, float | str]]:
    """Return a dashboard-friendly side-by-side metric table."""

    keys = ("f1", "precision", "recall", "fpr")
    return [
        {"metric": key, "logistic_regression": baseline[key], "world_model": world_model[key]}
        for key in keys
    ]


def probability_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Return classification and probability metrics at one operating threshold."""

    try:
        from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("probability metrics require scikit-learn") from exc
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    metrics = metrics_from_predictions(labels, (scores >= threshold).astype(np.int64))
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    except ValueError:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    tn, fp, fn, tp = confusion_matrix(labels, (scores >= threshold).astype(np.int64), labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray, minimum_recall: float = 0.0) -> float:
    """Choose the threshold with the best F1, optionally enforcing recall."""

    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.clip(scores, 0.0, 1.0))
    candidates = np.concatenate(([0.0], candidates, [0.5, 0.7, 0.9, 1.0]))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        metrics = probability_metrics(labels, scores, float(threshold))
        if metrics["recall"] >= minimum_recall and metrics["f1"] > best_f1:
            best_threshold, best_f1 = float(threshold), float(metrics["f1"])
    return best_threshold
