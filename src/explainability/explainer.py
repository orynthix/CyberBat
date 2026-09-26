"""Attention and SHAP explanation adapters for local inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def explain_features(predictor: Any, samples: np.ndarray, feature_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return sorted SHAP feature contributions for local model predictions."""

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP explanations require the optional 'shap' package") from exc
    values = shap.Explainer(
        predictor,
        masker=shap.maskers.Independent(np.asarray(samples, dtype=np.float32)),
    )(samples).values
    if values.ndim == 3:
        values = values.mean(axis=1)
    explanations = []
    for row in values:
        order = np.argsort(np.abs(row))[::-1]
        explanations.append(
            {
                "features": [
                    {"name": str(feature_names[index]), "contribution": float(row[index])}
                    for index in order
                ]
            }
        )
    return explanations


def summarize_alert(risk: float, attribution: dict[str, float]) -> str:
    """Create a concise human-readable alert summary."""

    top = sorted(attribution.items(), key=lambda item: abs(item[1]), reverse=True)[:2]
    drivers = " and ".join(f"{name} ({value:+.1%})" for name, value in top)
    return f"Infiltration risk {risk:.0%}" + (f" driven by {drivers}." if drivers else ".")
