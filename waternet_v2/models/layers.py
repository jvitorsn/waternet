"""Custom Keras layers for WaterNet v2.

ClampedLinear
    Enforces physical altitude constraints on model output.
    Predictions outside [alt_min, alt_max] are physically impossible and
    should not propagate during inference.
"""

from __future__ import annotations

import tensorflow as tf
from keras import layers


class ClampedLinear(layers.Layer):
    """Linear activation with hard clamping to the valid altitude range.

    After the regression head produces an unconstrained float, this layer
    clips it to [min_val, max_val] so predictions never violate physical
    limits (50 – 800 cm in the normalised [0, 1] domain that maps to
    50/800 – 800/800 after inverse-scaling).

    Args:
        min_val: Lower bound (in the same scale as the model output).
        max_val: Upper bound (in the same scale as the model output).
        **kwargs: Forwarded to ``tf.keras.layers.Layer``.
    """

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # noqa: D102
        return tf.clip_by_value(inputs, self.min_val, self.max_val)

    def get_config(self) -> dict:  # noqa: D102
        config = super().get_config()
        config.update({"min_val": self.min_val, "max_val": self.max_val})
        return config
