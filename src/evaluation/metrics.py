"""Full metrics suite for altitude regression (Phase 5 of the plan).

Metric selection rationale (Section 2.4.3 of the paper):

* **MAE** — optimal for Laplacian-distributed errors; robust to outliers.
* **RMSE** — optimal for Gaussian errors; quadratically penalises large errors
  (preferred for safety-critical altitude where large errors are dangerous).
* **MedAE** — outlier-robust central tendency.
* **R²** — fraction of variance explained; 1 = perfect, 0 = mean baseline.
* **MAPE** — scale-independent; a 10 cm error at 50 cm (20%) is more
  operationally significant than the same error at 500 cm (2%).
* **AbsRel** — absolute relative error; the Eigen et al. depth metric.
* **δ<1.25** — fraction of predictions within 25% of ground truth (threshold
  accuracy from the monocular depth estimation literature).

References:
    Hodson (2022, GMD) — RMSE vs MAE selection.
    Eigen et al. (2014, NIPS) — δ threshold accuracy for depth estimation.
    Rajapaksha et al. (2024) — accepted standard for altitude regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


# --------------------------------------------------------------------------- #
# Full metrics suite
# --------------------------------------------------------------------------- #

def compute_full_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute the full regression metrics suite.

    Args:
        y_true: Ground-truth altitudes in cm, shape (N,).
        y_pred: Predicted altitudes in cm, shape (N,).

    Returns:
        Ordered dictionary of metric name → float value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    ratio = np.maximum(y_pred / (y_true + 1e-8), y_true / (y_pred + 1e-8))

    return {
        "MAE (cm)":       float(mean_absolute_error(y_true, y_pred)),
        "RMSE (cm)":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE (cm)":     float(median_absolute_error(y_true, y_pred)),
        "R²":             float(r2_score(y_true, y_pred)),
        "MAPE (%)":       float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "Max Error (cm)": float(abs_errors.max()),
        "Error Std (cm)": float(errors.std()),
        "AbsRel":         float(np.mean(abs_errors / (y_true + 1e-8))),
        "δ<1.25 (%)":     float((ratio < 1.25).mean() * 100),
        "δ<1.25² (%)":    float((ratio < 1.5625).mean() * 100),
    }


def print_metrics(metrics: dict[str, float], title: str = "Metrics") -> None:
    """Pretty-print a metrics dictionary to stdout.

    Args:
        metrics: Output of ``compute_full_metrics``.
        title: Section header string.
    """
    width = 28
    print(f"\n{'─' * (width + 14)}")
    print(f"  {title}")
    print(f"{'─' * (width + 14)}")
    for name, value in metrics.items():
        print(f"  {name:<{width}} {value:>8.4f}")
    print(f"{'─' * (width + 14)}\n")


# --------------------------------------------------------------------------- #
# Stratified evaluation
# --------------------------------------------------------------------------- #

def evaluate_by_altitude_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Compute metrics stratified by altitude range.

    Reveals where the model fails (e.g., systematic underestimation at
    low altitudes where specular reflections are most dense).

    Args:
        y_true: Ground-truth altitudes in cm.
        y_pred: Predicted altitudes in cm.
        bins: Altitude bin edges in cm.  Defaults to
            [50, 100, 200, 300, 400, 500, 600, 700, 800].

    Returns:
        DataFrame with one row per altitude range and columns:
        range, n_samples, mae, rmse, mape, median_error, p95_error, r2.
    """
    if bins is None:
        bins = [50, 100, 200, 300, 400, 500, 600, 700, 800]

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    rows: list[dict] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        if not mask.any():
            continue
        yt, yp = y_true[mask], y_pred[mask]
        abs_err = np.abs(yt - yp)
        rows.append(
            {
                "range (cm)": f"{lo}–{hi}",
                "n_samples":  int(mask.sum()),
                "MAE":        float(mean_absolute_error(yt, yp)),
                "RMSE":       float(np.sqrt(mean_squared_error(yt, yp))),
                "MAPE (%)":   float(mean_absolute_percentage_error(yt, yp) * 100),
                "MedAE":      float(np.median(abs_err)),
                "P95 Error":  float(np.percentile(abs_err, 95)),
                "R²":         float(r2_score(yt, yp)),
            }
        )

    df_out = pd.DataFrame(rows)
    return df_out


# --------------------------------------------------------------------------- #
# Model comparison table
# --------------------------------------------------------------------------- #

def build_comparison_table(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Build an ablation study comparison table.

    Args:
        results: Dictionary mapping model name → ``(y_true, y_pred)`` tuples.

    Returns:
        DataFrame with one row per model and key metric columns.
    """
    rows: list[dict] = []
    for model_name, (y_true, y_pred) in results.items():
        m = compute_full_metrics(y_true, y_pred)
        rows.append(
            {
                "Model":      model_name,
                "MAE (cm)":   m["MAE (cm)"],
                "RMSE (cm)":  m["RMSE (cm)"],
                "R²":         m["R²"],
                "MAPE (%)":   m["MAPE (%)"],
                "δ<1.25 (%)": m["δ<1.25 (%)"],
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE (cm)").reset_index(drop=True)
