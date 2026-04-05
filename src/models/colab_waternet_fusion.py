# Ported from waternet_v2/models/multi_input.py
#
# Architecture (late fusion):
#
#   ┌─────────────────────────┐    ┌───────────────────────────┐
#   │  Image branch (CNN)     │    │  Feature branch (MLP)     │
#   │  V channel (H×W×1)      │    │  12-element feature vec   │
#   │  VGG-inspired backbone  │    │  Dense(64) → Dense(32)    │
#   │  → Dense(64) embedding  │    │  embedding                │
#   └────────────┬────────────┘    └──────────────┬────────────┘
#                │                                │
#                └──────── Concatenate ───────────┘
#                                 │
#                           Dense(128) → Dropout → Dense(1)
#
# Fusion strategy: late fusion — each modality is processed through its own
# independent stream and only the high-level representations are merged.
import keras
from keras import layers as KL

def build_waternet_fusion(
    input_shape: tuple = (224, 224, 1),
    n_features: int = 12,
    conv_filters=None,
    img_embed_units: int = 64,
    feat_dense_units:list = [64, 32],
    fusion_units: int = 128,
    dropout_rate: float = 0.3,
    apply_physical_constraint: bool = False,
    alt_min_norm: float = 0.0,
    alt_max_norm: float = 1.0,
):
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
        alt_min_norm: Lower bound for ClampedLinear.
        alt_max_norm: Upper bound for ClampedLinear.

    Returns:
        Uncompiled tf.keras.Model with inputs ["image_input", "feature_input"].
    """
    if conv_filters is None:
        conv_filters = [32, 64, 128, 256]
    if dense_units is None:
        dense_units = [512, 256, 128]

    # ── Image branch ─────────────────────────────────────────────────────── #
    inp = keras.Input(shape=input_shape, name="image_input")
    x   = inp

    for i, n_filters in enumerate(conv_filters):
        x = KL.Conv2D(n_filters, (5, 5), padding="same",
                      activation="relu", name=f"conv_{i}_a")(x)
        x = KL.Conv2D(n_filters, (5, 5), padding="same",
                      activation="relu", name=f"conv_{i}_b")(x)
        x = KL.MaxPool2D(name=f"pool_{i}")(x)
        if i >= 2:
            x = KL.Dropout(dropout_rate, name=f"drop_conv_{i}")(x)

    x = KL.GlobalAveragePooling2D(name="gap")(x)
    for i, units in enumerate(dense_units):
        x = KL.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = KL.Dropout(dropout_rate, name=f"drop_dense_{i}")(x)

    # ── Feature branch ────────────────────────────────────────────────────── #
    feat_inp = keras.Input(shape=(n_features,), name="feature_input")
    f = feat_inp
    for i, units in enumerate(feat_dense_units):
        f = KL.Dense(units, activation="relu", name=f"feat_dense{i}")(f)

    # ── Late fusion ───────────────────────────────────────────────────────── #
    merged = KL.Concatenate(name="fusion")([x, f])
    z = KL.Dense(fusion_units, activation="relu", name="fusion_dense")(merged)
    z = KL.Dropout(dropout_rate, name="fusion_dropout")(z)

    output = KL.Dense(1, activation="linear", name="output")(z)

    if apply_physical_constraint:
        output = ClampedLinear(
            min_val=alt_min_norm, max_val=alt_max_norm,
            name="physical_constraint",
        )(output)

    return keras.Model(
        inputs=[inp, feat_inp],
        outputs=output,
        name="WaterNet_Fusion",
    )