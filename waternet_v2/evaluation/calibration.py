"""Post-training bias calibration via cubic spline (Phase 4 of the plan).

Problem context (Section 4 of the implementation plan):
    The CNN exhibits non-linear systematic bias — it underestimates low
    altitudes (where dense specular reflections saturate the model) and
    overestimates intermediate heights.  A spline fitted on the validation
    set creates a monotonic mapping ``pred → calibrated_pred`` that absorbs
    this bias without retraining.

Algorithm:
    1. Bin validation predictions into 30 equal-frequency windows.
    2. For each bin with ≥ 10 samples compute (mean_pred, mean_true).
    3. Fit a cubic ``UnivariateSpline`` through (mean_pred → mean_true).
    4. Apply spline to test predictions; clip to [alt_min, alt_max].

Reference:
    Platt, J. (1999). Probabilistic outputs for support vector machines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline


class SplineCalibrator:
    """Cubic-spline post-training bias calibrator.

    Args:
        n_bins: Number of equal-width bins for computing bin-mean statistics.
        min_bin_samples: Minimum samples per bin required to include that
            bin in the spline fitting.
        smoothing: UnivariateSpline smoothing factor (relative to N).
        alt_min_cm: Physical lower altitude bound (50 cm).
        alt_max_cm: Physical upper altitude bound (800 cm).
    """

    def __init__(
        self,
        n_bins: int = 30,
        min_bin_samples: int = 10,
        smoothing: float = 0.1,
        alt_min_cm: float = 50.0,
        alt_max_cm: float = 800.0,
    ) -> None:
        self.n_bins = n_bins
        self.min_bin_samples = min_bin_samples
        self.smoothing = smoothing
        self.alt_min_cm = alt_min_cm
        self.alt_max_cm = alt_max_cm
        self._spline: UnivariateSpline | None = None
        self._fit_pred: np.ndarray | None = None
        self._fit_true: np.ndarray | None = None

    # ── Fitting ───────────────────────────────────────────────────────────── #

    def fit(self, val_pred: np.ndarray, val_true: np.ndarray) -> "SplineCalibrator":
        """Fit calibration spline on validation set predictions.

        Args:
            val_pred: Model predictions (cm) on the validation set.
            val_true: Ground-truth altitudes (cm) on the validation set.

        Returns:
            Self (for method chaining).
        """
        val_pred = np.asarray(val_pred, dtype=np.float64).ravel()
        val_true = np.asarray(val_true, dtype=np.float64).ravel()

        bins = np.linspace(val_pred.min(), val_pred.max(), self.n_bins + 1)
        bin_preds: list[float] = []
        bin_trues: list[float] = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (val_pred >= lo) & (val_pred < hi)
            if mask.sum() >= self.min_bin_samples:
                bin_preds.append(float(val_pred[mask].mean()))
                bin_trues.append(float(val_true[mask].mean()))

        if len(bin_preds) < 4:
            raise RuntimeError(
                f"Not enough bins with ≥{self.min_bin_samples} samples to fit spline "
                f"(got {len(bin_preds)}).  Try reducing min_bin_samples or n_bins."
            )

        self._fit_pred = np.array(bin_preds)
        self._fit_true = np.array(bin_trues)

        self._spline = UnivariateSpline(
            self._fit_pred,
            self._fit_true,
            s=self.smoothing * len(self._fit_pred),
            k=3,
            ext=3,   # extrapolate as boundary value (no wild swings)
        )

        print(
            f"[Calibration] Spline fitted on {len(bin_preds)} bins  "
            f"(min_samples={self.min_bin_samples})"
        )
        return self

    # ── Prediction ────────────────────────────────────────────────────────── #

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply spline calibration and clip to physical altitude range.

        Args:
            y_pred: Raw model predictions in cm.

        Returns:
            Calibrated predictions clipped to [alt_min_cm, alt_max_cm].

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self._spline is None:
            raise RuntimeError("Call fit() before transform().")

        calibrated = self._spline(np.asarray(y_pred, dtype=np.float64).ravel())
        return np.clip(calibrated, self.alt_min_cm, self.alt_max_cm).astype(np.float32)

    def fit_transform(
        self, val_pred: np.ndarray, val_true: np.ndarray
    ) -> np.ndarray:
        """Fit on validation data and return calibrated validation predictions.

        Args:
            val_pred: Validation predictions.
            val_true: Validation ground truth.

        Returns:
            Calibrated validation predictions.
        """
        self.fit(val_pred, val_true)
        return self.transform(val_pred)

    # ── Persistence ───────────────────────────────────────────────────────── #

    def save(self, path: str | Path) -> None:
        """Save the calibration control points to a JSON file.

        Args:
            path: Output file path (``*.json``).
        """
        if self._fit_pred is None:
            raise RuntimeError("Nothing to save; call fit() first.")

        data = {
            "fit_pred": self._fit_pred.tolist(),
            "fit_true": self._fit_true.tolist(),
            "n_bins": self.n_bins,
            "min_bin_samples": self.min_bin_samples,
            "smoothing": self.smoothing,
            "alt_min_cm": self.alt_min_cm,
            "alt_max_cm": self.alt_max_cm,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"[Calibration] Control points saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SplineCalibrator":
        """Reconstruct a calibrator from a previously saved JSON file.

        Args:
            path: Path to the JSON file written by ``save()``.

        Returns:
            Fitted ``SplineCalibrator`` instance.
        """
        with open(path) as fh:
            data = json.load(fh)

        obj = cls(
            n_bins=data["n_bins"],
            min_bin_samples=data["min_bin_samples"],
            smoothing=data["smoothing"],
            alt_min_cm=data["alt_min_cm"],
            alt_max_cm=data["alt_max_cm"],
        )
        obj._fit_pred = np.array(data["fit_pred"])
        obj._fit_true = np.array(data["fit_true"])
        obj._spline = UnivariateSpline(
            obj._fit_pred,
            obj._fit_true,
            s=obj.smoothing * len(obj._fit_pred),
            k=3,
            ext=3,
        )
        print(f"[Calibration] Loaded from {path}")
        return obj
