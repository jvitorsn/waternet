"""Lightweight VGG-inspired CNN — the existing baseline model (Phase 0 reference).

Architecture mirrors what was already in the project
(``src/water_distance_estimation.py``) but fixes the two critical issues:

1. Input is the explicit **HSV Value channel** (single-channel, float32 [0,1])
   instead of standard grayscale.
2. Target normalisation is done externally via ``StandardScaler``;
   this model outputs a raw float (no division by 800 embedded in the head).

The model is intentionally lightweight (~1.2 M parameters) to serve as
an ablation baseline compared to ResNet50 transfer learning.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from waternet_v2.models.layers import ClampedLinear


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    conv_filters: list[int] | None = None,
    dense_units: list[int] | None = None,
    dropout_rate: float = 0.3,
    apply_physical_constraint: bool = False,
    alt_min_norm: float = 0.0,
    alt_max_norm: float = 1.0,
) -> keras.Model:
    """Build the custom VGG-inspired regression CNN.

    Architecture overview:
        Conv(f, 5×5) → Conv(f, 5×5) → MaxPool   (×n_blocks)
        → GlobalAveragePooling → Dense(d₁) → Dropout
        → Dense(d₂) → Dropout → Dense(1, linear)

    Args:
        input_shape: (H, W, C) — should be (224, 224, 1) for the V channel.
        conv_filters: Number of filters per convolutional block.
            Defaults to [32, 64, 128, 256].
        dense_units: Units in each Dense regression layer.
            Defaults to [512, 256, 128].
        dropout_rate: Dropout probability after deeper conv blocks and
            dense layers.
        apply_physical_constraint: If True, clamp output to
            [alt_min_norm, alt_max_norm] via ``ClampedLinear``.
        alt_min_norm: Lower bound for ``ClampedLinear``.
        alt_max_norm: Upper bound for ``ClampedLinear``.

    Returns:
        Uncompiled ``tf.keras.Model``.
    """
    if conv_filters is None:
        conv_filters = [32, 64, 128, 256]
    if dense_units is None:
        dense_units = [512, 256, 128]

    inp = keras.Input(shape=input_shape, name="image_input")
    x = inp

    for i, n_filters in enumerate(conv_filters):
        x = layers.Conv2D(
            n_filters, (5, 5), padding="same", activation="relu",
            name=f"conv_{i}_a",
        )(x)
        x = layers.Conv2D(
            n_filters, (5, 5), padding="same", activation="relu",
            name=f"conv_{i}_b",
        )(x)
        x = layers.MaxPool2D(name=f"pool_{i}")(x)
        if i >= 2:
            x = layers.Dropout(dropout_rate, name=f"drop_conv_{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_dense_{i}")(x)

    output = layers.Dense(1, activation="linear", name="output")(x)

    if apply_physical_constraint:
        output = ClampedLinear(
            min_val=alt_min_norm, max_val=alt_max_norm, name="physical_constraint"
        )(output)

    return keras.Model(inputs=inp, outputs=output, name="WaterNet_CustomCNN")
