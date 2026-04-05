import keras
from keras import layers as KL
from .layers import ClampedLinear

def build_waternet(
    input_shape: tuple = (224, 224, 1),
    conv_filters=None,
    dense_units=None,
    dropout_rate: float = 0.3,
    apply_physical_constraint: bool = False,
    alt_min_norm: float = 0.0,
    alt_max_norm: float = 1.0,
) -> keras.Model:
    """Build the custom VGG-inspired regression CNN (ablation baseline).

    Architecture:
        Input (224, 224, 1)
        → [Conv2D(f,5×5) → Conv2D(f,5×5) → MaxPool2D] × 4   (conv blocks)
        → GlobalAveragePooling2D
        → Dense(512) → Dropout → Dense(256) → Dropout → Dense(128) → Dropout
        → Dense(1, linear)   [→ ClampedLinear if apply_physical_constraint]

    This model accepts only the image branch (no feature vector input).
    It is the ablation baseline before multi-input fusion is introduced.

    Args:
        input_shape: (H, W, C) — expects (224, 224, 1) for the V channel.
        conv_filters: Filters per convolutional block. Default [32,64,128,256].
        dense_units:  Units in each dense regression layer. Default [512,256,128].
        dropout_rate: Dropout probability applied after deeper layers.
        apply_physical_constraint: If True, clamp output via ClampedLinear.
        alt_min_norm: Lower clamp bound.
        alt_max_norm: Upper clamp bound.

    Returns:
        Uncompiled tf.keras.Model.
    """
    if conv_filters is None:
        conv_filters = [32, 64, 128, 256]
    if dense_units is None:
        dense_units = [512, 256, 128]

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

    output = KL.Dense(1, activation="linear", name="output")(x)

    if apply_physical_constraint:
        output = ClampedLinear(
            min_val=alt_min_norm, max_val=alt_max_norm,
            name="physical_constraint",
        )(output)

    return keras.Model(inputs=inp, outputs=output, name="WaterNet_CustomCNN")