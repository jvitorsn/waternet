"""Training callbacks for WaterNet v2 (Phase 3.2 of the implementation plan).

Standard callbacks provided:
  * ``build_standard_callbacks`` — assembles EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, and TensorBoard from configuration values.
  * ``EpochMetricsLogger`` — prints a concise metrics summary after each epoch.
  * ``PredictionHistoryCallback`` — records val predictions every N epochs so
    the bias evolution can be visualised post-training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from keras import callbacks


# --------------------------------------------------------------------------- #
# Standard callback bundle
# --------------------------------------------------------------------------- #

def build_standard_callbacks(
    checkpoint_dir: str | Path = "checkpoints",
    log_dir: str | Path = "logs",
    patience: int = 12,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    lr_min: float = 1e-7,
    monitor: str = "val_loss",
) -> list[callbacks.Callback]:
    """Create the standard callback stack for WaterNet training.

    Includes:
    1. ``EarlyStopping`` — restores best weights after *patience* stale epochs.
    2. ``ModelCheckpoint`` — saves the best model (by val_rmse) to disk.
    3. ``ReduceLROnPlateau`` — halves LR on val_loss plateau.
    4. ``TensorBoard`` — logs scalars and weight histograms.
    5. ``EpochMetricsLogger`` — console summary per epoch.

    Args:
        checkpoint_dir: Directory for model checkpoints.
        log_dir: Directory for TensorBoard event files.
        patience: EarlyStopping patience in epochs.
        lr_factor: Multiplicative LR reduction factor.
        lr_patience: Epochs of no val_loss improvement before LR reduction.
        lr_min: Minimum allowable learning rate.
        monitor: Metric to monitor for EarlyStopping and Checkpoint.

    Returns:
        List of Keras callback instances.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    early_stop = callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(
            Path(checkpoint_dir) / "model_{epoch:02d}_valloss{val_loss:.4f}.keras"
        ),
        monitor=monitor,
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr_min,
        verbose=1,
    )

    tensorboard = callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
    )

    return [early_stop, checkpoint, reduce_lr, tensorboard, EpochMetricsLogger()]


# --------------------------------------------------------------------------- #
# Custom callbacks
# --------------------------------------------------------------------------- #

class EpochMetricsLogger(callbacks.Callback):
    """Print a concise one-line metrics summary after each epoch."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if not logs:
            return
        parts = [f"Epoch {epoch + 1:03d}"]
        for key, value in logs.items():
            parts.append(f"{key}={value:.4f}")
        print("  ".join(parts))


class PredictionHistoryCallback(callbacks.Callback):
    """Record full validation predictions at specified epoch intervals.

    Useful for tracking the evolution of systematic bias over training.

    Args:
        val_data: Tuple ``(X_val, y_val_true)`` in original scale.
        record_every: Record predictions every this many epochs.
        scaler: Optional sklearn scaler to inverse-transform model output.
    """

    def __init__(
        self,
        val_data: tuple,
        record_every: int = 5,
        scaler: object | None = None,
    ) -> None:
        super().__init__()
        self.X_val, self.y_val_true = val_data
        self.record_every = record_every
        self.scaler = scaler
        self.history: dict[int, np.ndarray] = {}

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if (epoch + 1) % self.record_every != 0:
            return
        preds = self.model.predict(self.X_val, verbose=0).ravel()
        if self.scaler is not None:
            preds = self.scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        self.history[epoch + 1] = preds
