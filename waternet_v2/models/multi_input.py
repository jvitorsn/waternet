"""Multi-input model with late fusion of visual and handcrafted features (Phase 2.3).

Architecture (Section 2.4.1, Multi-input CNN paragraph):

    ┌─────────────────────────┐    ┌───────────────────────────┐
    │  Image branch (CNN)     │    │  Feature branch (MLP)     │
    │  V channel (H×W×1)      │    │  12-element feature vec   │
    │  VGG-inspired backbone  │    │  Dense(64) → Dense(32)    │
    │  → Dense(64) embedding  │    │  embedding                │
    └────────────┬────────────┘    └──────────────┬────────────┘
                 │                                │
                 └──────── Concatenate ───────────┘
                                  │
                            Dense(64) → Dropout → Dense(1)

Fusion strategy: **late fusion** — each modality is processed through its
own independent stream and only the high-level representations are merged.
This is the simplest and most robust approach, and avoids forcing the CNN to
process raw numerical tabular features in its early layers.

Reference: Wolf et al. (DAFT, 2022) for advanced intermediate fusion if needed.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from waternet_v2.models.layers import ClampedLinear


def build_multi_input_model(
    img_shape: tuple[int, int, int] = (224, 224, 1),
    n_features: int = 12,
    conv_filters: list[int] | None = None,
    img_embed_units: int = 64,
    feat_dense_units: list[int] | None = None,
    fusion_units: int = 128,
    dropout_rate: float = 0.3,
    apply_physical_constraint: bool = False,
    alt_min_norm: float = 0.0,
    alt_max_norm: float = 1.0,
) -> keras.Model:
    """Build the multi-input late-fusion altitude regression model.

    Args:
        img_shape: Image input shape (H, W, 1).
        n_features: Length of the handcrafted feature vector (default 12).
        conv_filters: Conv block filter counts for the image branch.
            Defaults to [32, 64, 128].
        img_embed_units: Dense units producing the image embedding.
        feat_dense_units: Dense units in the feature branch.
            Defaults to [64, 32].
        fusion_units: Dense units in the merged fusion layer.
        dropout_rate: Dropout probability in the regression head.
        apply_physical_constraint: Clamp output to [alt_min_norm, alt_max_norm].
        alt_min_norm: Lower bound for ``ClampedLinear``.
        alt_max_norm: Upper bound for ``ClampedLinear``.

    Returns:
        Uncompiled ``tf.keras.Model`` with inputs
        ``["image_input", "feature_input"]``.
    """
    if conv_filters is None:
        conv_filters = [32, 64, 128]
    if feat_dense_units is None:
        feat_dense_units = [64, 32]

    # ── Image branch ──────────────────────────────────────────────────────── #
    img_inp = keras.Input(shape=img_shape, name="image_input")
    x = img_inp
    for i, n_filt in enumerate(conv_filters):
        x = layers.Conv2D(n_filt, 3, padding="same", activation="relu",
                          name=f"img_conv{i}_a")(x)
        x = layers.Conv2D(n_filt, 3, padding="same", activation="relu",
                          name=f"img_conv{i}_b")(x)
        x = layers.MaxPool2D(name=f"img_pool{i}")(x)

    x = layers.GlobalAveragePooling2D(name="img_gap")(x)
    x = layers.Dense(img_embed_units, activation="relu",
                     name="img_embedding")(x)

    # ── Feature branch ────────────────────────────────────────────────────── #
    feat_inp = keras.Input(shape=(n_features,), name="feature_input")
    f = feat_inp
    for i, units in enumerate(feat_dense_units):
        f = layers.Dense(units, activation="relu", name=f"feat_dense{i}")(f)

    # ── Late fusion ───────────────────────────────────────────────────────── #
    merged = layers.Concatenate(name="fusion")([x, f])
    z = layers.Dense(fusion_units, activation="relu", name="fusion_dense")(merged)
    z = layers.Dropout(dropout_rate, name="fusion_dropout")(z)
    output = layers.Dense(1, activation="linear", name="output")(z)

    if apply_physical_constraint:
        output = ClampedLinear(
            min_val=alt_min_norm, max_val=alt_max_norm,
            name="physical_constraint"
        )(output)

    return keras.Model(
        inputs=[img_inp, feat_inp],
        outputs=output,
        name="WaterNet_MultiInput",
    )


# --------------------------------------------------------------------------- #
# ResNet50 multi-input variant
# --------------------------------------------------------------------------- #

def build_resnet50_multi_input(
    img_shape: tuple[int, int, int] = (224, 224, 1),
    n_features: int = 12,
    freeze_backbone: bool = True,
    feat_dense_units: list[int] | None = None,
    fusion_units: int = 128,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Multi-input model with ResNet50 backbone in the image branch.

    Uses the same channel-replication trick as ``build_resnet50_altitude_model``
    to handle single-channel input.

    Args:
        img_shape: Image input shape (H, W, 1).
        n_features: Handcrafted feature vector length.
        freeze_backbone: Freeze backbone during Stage 1 training.
        feat_dense_units: Dense units in feature branch; defaults to [64, 32].
        fusion_units: Dense units after fusion.
        dropout_rate: Dropout in the head.

    Returns:
        Uncompiled ``tf.keras.Model``.
    """
    if feat_dense_units is None:
        feat_dense_units = [64, 32]

    # ── Image branch (ResNet50) ───────────────────────────────────────────── #
    img_inp = keras.Input(shape=img_shape, name="image_input")
    x = layers.Concatenate(name="channel_replicate")([img_inp, img_inp, img_inp])
    x = tf.keras.applications.resnet50.preprocess_input(x)

    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_shape[0], img_shape[1], 3),
    )
    backbone.trainable = not freeze_backbone
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="img_gap")(x)
    x = layers.Dense(128, activation="relu", name="img_embedding")(x)

    # ── Feature branch ────────────────────────────────────────────────────── #
    feat_inp = keras.Input(shape=(n_features,), name="feature_input")
    f = feat_inp
    for i, units in enumerate(feat_dense_units):
        f = layers.Dense(units, activation="relu", name=f"feat_dense{i}")(f)

    # ── Late fusion ───────────────────────────────────────────────────────── #
    merged = layers.Concatenate(name="fusion")([x, f])
    z = layers.Dense(fusion_units, activation="relu", name="fusion_dense")(merged)
    z = layers.Dropout(dropout_rate, name="fusion_dropout")(z)
    output = layers.Dense(1, activation="linear", name="output")(z)

    return keras.Model(
        inputs=[img_inp, feat_inp],
        outputs=output,
        name="WaterNet_ResNet50_MultiInput",
    )
