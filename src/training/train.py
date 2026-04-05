"""Main training pipeline for WaterNet v2 (Phase 3 of the implementation plan).

Key design decisions implemented here:

* **Target normalisation** (Phase 0.2): ``StandardScaler`` is fit on the
  training targets and applied to val/test.  The scaler parameters are
  saved alongside the model for correct inference.

* **AdamW + Huber loss** (Phase 3.1): AdamW with weight decay and gradient
  clipping avoids exploding gradients; Huber loss (δ=1.0) combines MSE's
  smooth gradient near zero with MAE's robustness to outlier altitudes.

* **Cosine-decay LR with warm-up** via ``ReduceLROnPlateau`` fallback.

Usage example::

    from waternet_v2.training.train import run_training_pipeline

    results = run_training_pipeline(
        model_type="multi_input",   # "custom_cnn" | "resnet50" | "multi_input"
        config_path=None,           # uses default.yaml
    )
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras

from waternet_v2.configs import load_config
from waternet_v2.data.augmentation import WaterAugmenter
from waternet_v2.data.dataset import WaterDataSequence, make_stratified_split
from waternet_v2.data.preprocessing import extract_feature_vector, load_and_extract_value_channel
from waternet_v2.models.custom_cnn import build_custom_cnn
from waternet_v2.models.multi_input import build_multi_input_model
from waternet_v2.models.resnet_baseline import build_resnet50_altitude_model
from waternet_v2.training.callbacks import build_standard_callbacks


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #

def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    clip_norm: float = 1.0,
    huber_delta: float = 1.0,
) -> None:
    """Compile *model* in-place with AdamW + Huber loss + standard metrics.

    Args:
        model: Uncompiled Keras model.
        learning_rate: Initial learning rate.
        weight_decay: AdamW weight decay coefficient.
        clip_norm: Gradient clipping norm.
        huber_delta: Huber loss transition point.
    """
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clip_norm,
        ),
        loss=keras.losses.Huber(delta=huber_delta),
        metrics=[
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    print(f"[Compile] lr={learning_rate}  weight_decay={weight_decay}  "
          f"huber_delta={huber_delta}")
    print(f"[Compile] Trainable params: "
          f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}")


# --------------------------------------------------------------------------- #
# Pre-compute feature matrix
# --------------------------------------------------------------------------- #

def precompute_features(
    df: pd.DataFrame,
    image_dir: str | Path,
    target_size: tuple[int, int] = (224, 224),
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-extract V channel images and feature vectors for an entire split.

    Returns arrays suitable for direct NumPy / LightGBM training, avoiding
    the per-batch overhead of on-the-fly feature extraction.

    Args:
        df: DataFrame with columns ``nome`` and ``distancia``.
        image_dir: Directory containing image files.
        target_size: (width, height) resize target.
        verbose: Print progress every 1000 samples.

    Returns:
        ``(images, features)`` where:
        - ``images`` shape ``(N, H, W, 1)``, float32
        - ``features`` shape ``(N, 12)``, float32
    """
    image_dir = Path(image_dir)
    images: list[np.ndarray] = []
    features: list[np.ndarray] = []

    for i, row in df.iterrows():
        if verbose and i % 1000 == 0:
            print(f"  Extracting features: {i}/{len(df)}", end="\r")

        v = load_and_extract_value_channel(
            str(image_dir / row["nome"]), target_size
        )
        images.append(v[..., np.newaxis])
        features.append(extract_feature_vector(v))

    if verbose:
        print(f"  Feature extraction complete: {len(df)} samples.")

    return np.array(images, dtype=np.float32), np.array(features, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def run_training_pipeline(
    model_type: str = "multi_input",
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs",
    seed: int = 42,
) -> dict[str, Any]:
    """Run the complete WaterNet v2 training pipeline.

    Steps:
    1. Load config and set seeds.
    2. Load CSV and perform stratified 70/15/15 split.
    3. Fit StandardScaler on training targets.
    4. Pre-extract images + feature vectors for all splits.
    5. Build and compile model.
    6. Train with standard callbacks.
    7. Save model, scaler, and training history.

    Args:
        model_type: One of ``"custom_cnn"``, ``"resnet50"``, ``"multi_input"``.
        config_path: Optional path to override config YAML.
        output_dir: Directory to save model artefacts.
        seed: Global random seed.

    Returns:
        Dictionary with keys:
        ``model``, ``history``, ``scaler_y``,
        ``splits`` (dict of DataFrames), ``config``.
    """
    # ── Setup ────────────────────────────────────────────────────────────── #
    import os
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    img_size = tuple(data_cfg["image_size"])  # (H, W)
    target_size_wh = (img_size[1], img_size[0])  # OpenCV uses (W, H)

    print(f"[Pipeline] Model type : {model_type}")
    print(f"[Pipeline] Image size : {img_size}")
    print(f"[Pipeline] Output dir : {out_dir}")

    # ── Load data ────────────────────────────────────────────────────────── #
    df = pd.read_csv(data_cfg["csv_path"])
    print(f"[Pipeline] Dataset loaded: {len(df):,} rows")

    df_train, df_val, df_test = make_stratified_split(
        df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        n_bins=data_cfg["n_bins_stratify"],
        seed=seed,
    )

    # ── Target normalisation (Phase 0.2) ─────────────────────────────────── #
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(
        df_train[["distancia"]].values
    ).ravel().astype(np.float32)
    y_val = scaler_y.transform(
        df_val[["distancia"]].values
    ).ravel().astype(np.float32)

    print(
        f"[Scaler]   mean={scaler_y.mean_[0]:.1f} cm  "
        f"std={scaler_y.scale_[0]:.1f} cm"
    )

    # ── Build model ──────────────────────────────────────────────────────── #
    h, w = img_size
    if model_type == "custom_cnn":
        model = build_custom_cnn(input_shape=(h, w, 1))
    elif model_type == "resnet50":
        model = build_resnet50_altitude_model(input_shape=(h, w, 1))
    elif model_type == "multi_input":
        model = build_multi_input_model(
            img_shape=(h, w, 1),
            n_features=cfg["features"]["n_features"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model.summary(print_fn=print)

    compile_model(
        model,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        clip_norm=train_cfg["clip_norm"],
        huber_delta=train_cfg["huber_delta"],
    )

    # ── Data generators ──────────────────────────────────────────────────── #
    augmenter = WaterAugmenter(seed=seed)
    image_dir = data_cfg["image_dir"]

    train_gen = WaterDataSequence(
        df_train, image_dir, y_train,
        target_size=target_size_wh,
        batch_size=train_cfg["batch_size"],
        augmenter=augmenter,
        training=True,
    )
    val_gen = WaterDataSequence(
        df_val, image_dir, y_val,
        target_size=target_size_wh,
        batch_size=train_cfg["batch_size"],
        training=False,
        shuffle=False,
    )

    # ── Train ────────────────────────────────────────────────────────────── #
    cbs = build_standard_callbacks(
        checkpoint_dir=out_dir / "checkpoints",
        log_dir=out_dir / "logs",
        patience=train_cfg["patience"],
        lr_factor=train_cfg["lr_factor"],
        lr_patience=train_cfg["lr_patience"],
        lr_min=train_cfg["lr_min"],
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=train_cfg["epochs"],
        callbacks=cbs,
        verbose=1,
    )

    # ── Persist artefacts ────────────────────────────────────────────────── #
    model.save(out_dir / "waternet_v2_final.keras")

    with open(out_dir / "scaler_y.pkl", "wb") as fh:
        pickle.dump(scaler_y, fh)

    scaler_meta = {
        "mean": float(scaler_y.mean_[0]),
        "scale": float(scaler_y.scale_[0]),
    }
    with open(out_dir / "scaler_y.json", "w") as fh:
        json.dump(scaler_meta, fh, indent=2)

    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(out_dir / "training_history.json", "w") as fh:
        json.dump(history_dict, fh, indent=2)

    print(f"\n[Pipeline] Artefacts saved to: {out_dir}")

    return {
        "model": model,
        "history": history.history,
        "scaler_y": scaler_y,
        "splits": {"train": df_train, "val": df_val, "test": df_test},
        "config": cfg,
    }
