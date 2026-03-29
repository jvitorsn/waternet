"""Publication-quality visualisations for altitude regression (Phase 5.3).

Seven plot types required by the paper (Section 5.3 of the plan):
  1. Training curves — loss + RMSE per epoch (train + val).
  2. Scatter plot — predictions vs ground truth with identity line.
  3. Residual plot — error vs ground truth to show systematic bias.
  4. Box plot — error distribution per altitude range.
  5. Confusion matrix heatmap — when altitude is discretised into classes.
  6. Architecture comparison — grouped bar chart (ablation study).
  7. Error histogram — distribution with fitted Gaussian.

All plots use the ``seaborn`` academic theme and are returned as
``matplotlib.figure.Figure`` objects so callers can save or display them.

Also includes:
  * ``plot_fft_altitude_relationship`` — shows how FFT energy shifts with altitude.
  * ``plot_gradcam_overlay`` — Grad-CAM heatmap over input image.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

# ── Academic plot style ───────────────────────────────────────────────────── #
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Serif",
    }
)

_PALETTE = "muted"
_FIGSIZE = (8, 6)


# --------------------------------------------------------------------------- #
# 1. Training curves
# --------------------------------------------------------------------------- #

def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "WaterNet v2 — Training Curves",
) -> Figure:
    """Plot loss and RMSE training + validation curves vs epoch.

    Args:
        history: Keras ``history.history`` dictionary.
        title: Figure title.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, y=1.01)

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["loss"], label="Train Loss", lw=1.8)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss",
                     lw=1.8, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # RMSE
    rmse_key = "rmse" if "rmse" in history else "root_mean_squared_error"
    val_rmse_key = "val_rmse" if "val_rmse" in history else "val_root_mean_squared_error"

    if rmse_key in history:
        axes[1].plot(epochs, history[rmse_key], label="Train RMSE", lw=1.8)
    if val_rmse_key in history:
        axes[1].plot(epochs, history[val_rmse_key], label="Val RMSE",
                     lw=1.8, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (normalised)")
    axes[1].set_title("RMSE")
    axes[1].legend()

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. Scatter: predictions vs ground truth
# --------------------------------------------------------------------------- #

def plot_scatter_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "WaterNet",
    alt_min: float = 50.0,
    alt_max: float = 800.0,
) -> Figure:
    """Scatter plot of predicted vs true altitudes with identity line.

    Points are coloured by altitude range to reveal range-dependent bias.

    Args:
        y_true: Ground-truth altitudes in cm.
        y_pred: Predicted altitudes in cm.
        model_name: Label for the title.
        alt_min: Minimum altitude for axis limits.
        alt_max: Maximum altitude for axis limits.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    # Colour by altitude range
    bins = [50, 200, 400, 600, 800]
    colours = sns.color_palette(_PALETTE, n_colors=len(bins) - 1)
    labels = [f"{lo}–{hi} cm" for lo, hi in zip(bins[:-1], bins[1:])]

    for (lo, hi), colour, label in zip(
        zip(bins[:-1], bins[1:]), colours, labels
    ):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            ax.scatter(
                y_true[mask], y_pred[mask],
                s=12, alpha=0.5, color=colour, label=label,
            )

    # Identity line
    lims = [alt_min, alt_max]
    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")

    # Linear regression trend
    slope, intercept, r, *_ = stats.linregress(y_true, y_pred)
    x_trend = np.array(lims)
    ax.plot(x_trend, slope * x_trend + intercept, "r-", lw=1.2, alpha=0.7,
            label=f"Linear fit (r={r:.3f})")

    ax.set_xlim(alt_min, alt_max)
    ax.set_ylim(alt_min, alt_max)
    ax.set_xlabel("True Altitude (cm)")
    ax.set_ylabel("Predicted Altitude (cm)")
    ax.set_title(f"{model_name} — Predictions vs Ground Truth")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. Residual plot
# --------------------------------------------------------------------------- #

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "WaterNet",
) -> Figure:
    """Error vs ground truth to reveal systematic bias patterns.

    Args:
        y_true: Ground-truth altitudes in cm.
        y_pred: Predicted altitudes in cm.
        model_name: Label for the title.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    errors = y_pred - y_true   # positive = over-estimate
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    ax.scatter(y_true, errors, s=10, alpha=0.4,
               color=sns.color_palette(_PALETTE)[0])
    ax.axhline(0, color="k", linestyle="--", lw=1.5, label="Zero error")
    ax.axhline(errors.mean(), color="r", linestyle="-",
               lw=1.2, label=f"Mean bias = {errors.mean():.1f} cm")

    ax.set_xlabel("True Altitude (cm)")
    ax.set_ylabel("Error (pred − true)  [cm]")
    ax.set_title(f"{model_name} — Residual Plot")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. Box plot by altitude range
# --------------------------------------------------------------------------- #

def plot_error_boxplot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "WaterNet",
    bins: list[float] | None = None,
) -> Figure:
    """Box plot of absolute errors stratified by altitude range.

    Args:
        y_true: Ground-truth altitudes in cm.
        y_pred: Predicted altitudes in cm.
        model_name: Label for the title.
        bins: Altitude bin edges; defaults to [50,100,200,400,600,800].

    Returns:
        ``matplotlib.figure.Figure``.
    """
    if bins is None:
        bins = [50, 100, 200, 400, 600, 800]

    import pandas as pd
    abs_errors = np.abs(y_pred - y_true)
    bin_labels = [f"{lo}–{hi}" for lo, hi in zip(bins[:-1], bins[1:])]
    bin_idx = np.digitize(y_true, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_labels) - 1)

    df_plot = pd.DataFrame(
        {"Altitude Range (cm)": [bin_labels[i] for i in bin_idx],
         "Absolute Error (cm)": abs_errors}
    )

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.boxplot(
        data=df_plot, x="Altitude Range (cm)", y="Absolute Error (cm)",
        palette=_PALETTE, ax=ax,
    )
    ax.set_title(f"{model_name} — Error Distribution by Altitude Range")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. Architecture comparison (ablation study)
# --------------------------------------------------------------------------- #

def plot_model_comparison(
    comparison_df,
    metrics: list[str] | None = None,
) -> Figure:
    """Grouped bar chart comparing multiple models on key metrics.

    Args:
        comparison_df: DataFrame from ``build_comparison_table()``.
        metrics: Metric columns to plot; defaults to
            ``["MAE (cm)", "RMSE (cm)"]``.

    Returns:
        ``matplotlib.figure.Figure``.
    """

    if metrics is None:
        metrics = ["MAE (cm)", "RMSE (cm)"]

    melted = comparison_df.melt(
        id_vars="Model", value_vars=metrics,
        var_name="Metric", value_name="Value",
    )

    fig, ax = plt.subplots(figsize=(max(8, len(comparison_df) * 1.5), 5))
    sns.barplot(
        data=melted, x="Model", y="Value", hue="Metric",
        palette="Set2", ax=ax,
    )
    ax.set_title("WaterNet v2 — Ablation Study")
    ax.set_ylabel("Error (cm)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 7. Error histogram
# --------------------------------------------------------------------------- #

def plot_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "WaterNet",
    bins: int = 50,
) -> Figure:
    """Histogram of prediction errors with fitted Gaussian overlay.

    Args:
        y_true: Ground-truth altitudes in cm.
        y_pred: Predicted altitudes in cm.
        model_name: Label for title.
        bins: Number of histogram bins.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    errors = y_pred - y_true
    mu, sigma = errors.mean(), errors.std()

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(errors, bins=bins, density=True, alpha=0.6,
            color=sns.color_palette(_PALETTE)[2], label="Error distribution")

    x_fit = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma), "r-", lw=2,
            label=f"Gaussian fit\n(μ={mu:.1f}, σ={sigma:.1f})")

    ax.axvline(0, color="k", linestyle="--", lw=1.2)
    ax.set_xlabel("Prediction Error (cm)")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Error Histogram")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# FFT altitude relationship
# --------------------------------------------------------------------------- #

def plot_fft_altitude_relationship(
    sample_images: dict[int, np.ndarray],
) -> Figure:
    """Show how FFT magnitude spectrum changes with altitude.

    Args:
        sample_images: Dict mapping altitude (int, cm) → V channel array
            (float32, [0,1]).

    Returns:
        ``matplotlib.figure.Figure``.
    """
    from waternet_v2.data.preprocessing import compute_fft_magnitude

    altitudes = sorted(sample_images.keys())
    n = len(altitudes)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, alt in enumerate(altitudes):
        v = sample_images[alt]
        fft_mag = compute_fft_magnitude(v)

        axes[0, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{alt} cm", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(fft_mag, cmap="inferno", vmin=0, vmax=1)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("V channel", fontsize=9)
    axes[1, 0].set_ylabel("FFT magnitude", fontsize=9)
    fig.suptitle("FFT Frequency Content vs Altitude", fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Grad-CAM overlay
# --------------------------------------------------------------------------- #

def plot_gradcam_overlay(
    v_channel: np.ndarray,
    cam_heatmap: np.ndarray,
    true_alt: float,
    pred_alt: float,
) -> Figure:
    """Overlay a Grad-CAM heatmap on the input V channel.

    Args:
        v_channel: Float32 input image in [0, 1], shape (H, W).
        cam_heatmap: Float32 normalised heatmap in [0, 1], shape (H, W).
        true_alt: Ground-truth altitude in cm.
        pred_alt: Predicted altitude in cm.

    Returns:
        ``matplotlib.figure.Figure``.
    """

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original
    axes[0].imshow(v_channel, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input (V channel)")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(cam_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    v_rgb = np.stack([v_channel] * 3, axis=-1)
    heatmap_rgb = plt.cm.jet(cam_heatmap)[:, :, :3]
    overlay = 0.55 * v_rgb + 0.45 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nTrue: {true_alt:.0f} cm  |  Pred: {pred_alt:.0f} cm"
    )
    axes[2].axis("off")

    fig.tight_layout()
    return fig
