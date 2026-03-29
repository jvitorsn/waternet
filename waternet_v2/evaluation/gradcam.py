"""Grad-CAM explainability for the WaterNet CNN (Section 2.4.3 of the paper).

Grad-CAM (Gradient-weighted Class Activation Mapping) reveals which spatial
regions of the input image drive the altitude prediction.  For a regression
model the "class score" is the scalar predicted altitude.

Algorithm (Selvaraju et al., 2017 — Eq. 2.20–2.22 of the paper):

    1. Forward pass: record activations A^k of the last conv layer.
    2. Compute gradients ∂y/∂A^k  where y = predicted altitude.
    3. Global-average-pool the gradients: α_k = (1/Z) Σ_ij ∂y/∂A^k_ij
    4. Weighted sum + ReLU: L = ReLU(Σ_k α_k · A^k)
    5. Upsample L to input spatial dimensions.

Physical expectation: the heatmap should highlight regions of wave texture
and specular reflections — NOT image borders or constant-intensity areas —
confirming that the network attends to altitude-relevant features.

References:
    Selvaraju et al. (2017) ICCV — Grad-CAM.
    Chattopadhyay et al. (2018) — Grad-CAM++.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import cv2
import keras


def compute_gradcam(
    model: keras.Model,
    v_channel: np.ndarray,
    last_conv_layer_name: str,
    use_features: bool = False,
    feature_vector: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single V-channel input.

    Args:
        model: Compiled Keras regression model.
        v_channel: Single-channel float32 image in [0, 1], shape (H, W).
        last_conv_layer_name: Name of the last convolutional layer whose
            feature maps are used (e.g. ``"img_conv2_b"`` or ``"conv5_block3_out"``).
        use_features: Whether the model expects a feature vector input
            (``True`` for the multi-input architecture).
        feature_vector: Required when ``use_features=True``.

    Returns:
        Normalised Grad-CAM heatmap, float32 in [0, 1], shape (H, W).

    Raises:
        ValueError: If the named layer is not found in the model.
    """
    # Validate layer name
    layer_names = [layer.name for layer in model.layers]
    if last_conv_layer_name not in layer_names:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found. "
            f"Available layers: {[n for n in layer_names if 'conv' in n.lower()]}"
        )

    # Build gradient model: outputs [conv_activations, prediction]
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    # Prepare input tensors
    img_tensor = tf.constant(
        v_channel[np.newaxis, ..., np.newaxis].astype(np.float32)
    )

    if use_features and feature_vector is not None:
        inputs = [img_tensor,
                  tf.constant(feature_vector[np.newaxis].astype(np.float32))]
    else:
        inputs = img_tensor

    # Forward + gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        altitude_score = predictions[0, 0]   # scalar regression output

    grads = tape.gradient(altitude_score, conv_outputs)

    # α_k = global average of ∂y/∂A^k
    alpha_k = tf.reduce_mean(grads, axis=(0, 1, 2))   # shape: (C,)

    # Weighted sum of activation maps + ReLU
    conv_out = conv_outputs[0]                          # shape: (h, w, C)
    cam = tf.reduce_sum(alpha_k * conv_out, axis=-1)   # shape: (h, w)
    cam = tf.nn.relu(cam).numpy().astype(np.float32)

    # Normalise
    max_val = cam.max()
    if max_val > 1e-8:
        cam /= max_val

    # Upsample to input spatial resolution
    h, w = v_channel.shape
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    return cam_resized.astype(np.float32)


def find_last_conv_layer(model: keras.Model) -> str:
    """Find the name of the last Conv2D layer in a model.

    Args:
        model: Keras model to inspect.

    Returns:
        Name of the last Conv2D layer.

    Raises:
        RuntimeError: If no Conv2D layer is found.
    """
    last_conv_name: str | None = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
        # Handle sub-models (e.g. ResNet50 backbone)
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    last_conv_name = sublayer.name

    if last_conv_name is None:
        raise RuntimeError("No Conv2D layer found in the model.")

    return last_conv_name
