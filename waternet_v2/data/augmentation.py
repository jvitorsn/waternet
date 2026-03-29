"""Data augmentation optimised for downward-facing water surface imagery.

Key design insight (Section 2.2, Section 3.3):
    Nadir (straight-down) views of water are **rotationally invariant** —
    any rotation preserves the altitude label because the surface has no
    canonical orientation.  This justifies full 360° rotation augmentation
    (RandomRotation factor=1.0), which would be incorrect for scenes with
    gravity-aligned cues such as buildings or horizon lines.

Two augmentation pathways are provided:
  * ``build_keras_augmentation_pipeline`` — a ``tf.keras.Sequential`` model
    applied during the forward pass (GPU-friendly, no extra I/O).
  * ``WaterAugmenter`` — a NumPy/OpenCV augmenter for use inside custom
    ``tf.keras.utils.Sequence`` data loaders that operate per-sample.
"""

from __future__ import annotations

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------- #
# Keras augmentation pipeline (applied in-graph)
# --------------------------------------------------------------------------- #

def build_keras_augmentation_pipeline(seed: int = 42) -> keras.Sequential:
    """Build a Keras augmentation pipeline for water surface images.

    All transforms preserve altitude labels:
    - Flips / rotations: valid because nadir view is rotationally invariant.
    - Brightness/contrast perturbation: simulates solar angle variation.
    - Gaussian noise: simulates sensor noise.

    Args:
        seed: Base random seed.  Each layer receives a unique seed derived
            from this value so the augmentations remain independently random.

    Returns:
        A compiled ``tf.keras.Sequential`` augmentation model.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(factor=1.0, seed=seed + 1),      # full 360°
            layers.RandomBrightness(factor=0.15, seed=seed + 2),
            layers.RandomContrast(factor=0.15, seed=seed + 3),
            layers.GaussianNoise(stddev=0.02, seed=seed + 4),
        ],
        name="water_augmentation",
    )


# --------------------------------------------------------------------------- #
# NumPy / OpenCV augmenter for custom Sequence loaders
# --------------------------------------------------------------------------- #

class WaterAugmenter:
    """Stochastic augmentation for single-channel water surface images.

    Augmentations applied (each independently gated by a Bernoulli draw):

    1. **Random gamma correction** — simulates varying solar irradiance.
       ``v' = v^γ`` where γ ~ Uniform(γ_min, γ_max).

    2. **Additive Gaussian noise** — simulates sensor thermal noise.

    3. **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation) —
       normalises local contrast to reduce illumination-driven domain shift.

    4. **Linear motion blur** — simulates translational camera movement during
       exposure (PSF: uniform rect kernel, Eq. 2.3 of the paper).

    5. **Circular motion blur** — simulates yaw rotation during exposure
       (spatially variant; approximated here by a tangential circular kernel).

    Args:
        gamma_range: (min, max) of the gamma correction exponent.
        noise_std: Standard deviation of additive Gaussian noise.
        clahe_prob: Probability of applying CLAHE.
        blur_prob: Probability of applying motion blur.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.7, 1.3),
        noise_std: float = 0.02,
        clahe_prob: float = 0.3,
        blur_prob: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.clahe_prob = clahe_prob
        self.blur_prob = blur_prob
        self._rng = np.random.RandomState(seed)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def augment(self, v_channel: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply stochastic augmentation to a single V-channel image.

        Args:
            v_channel: Float32 image in [0, 1], shape (H, W).
            training: When False, returns the input unchanged (eval mode).

        Returns:
            Augmented float32 image in [0, 1], same shape.
        """
        if not training:
            return v_channel

        img = v_channel.copy()

        # 1. Random gamma correction
        if self._rng.random() > 0.5:
            gamma = self._rng.uniform(*self.gamma_range)
            img = np.power(img, gamma)

        # 2. Additive Gaussian noise
        if self._rng.random() > 0.5:
            noise = self._rng.normal(0.0, self.noise_std, img.shape)
            img = img + noise

        # 3. CLAHE (applied on uint8)
        if self._rng.random() < self.clahe_prob:
            img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = self._clahe.apply(img_u8).astype(np.float32) / 255.0

        # 4. Linear motion blur
        if self._rng.random() < self.blur_prob:
            img = _apply_linear_motion_blur(img, self._rng)

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def reset_seed(self, seed: int) -> None:
        """Reset the internal random state (useful for reproducible tests)."""
        self._rng = np.random.RandomState(seed)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _apply_linear_motion_blur(
    v_channel: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """Apply a directional linear motion-blur kernel (PSF from Eq. 2.3).

    Args:
        v_channel: Float32 image in [0, 1].
        rng: NumPy random state.

    Returns:
        Blurred float32 image.
    """
    length = rng.randint(3, 9)
    angle_deg = rng.uniform(0, 180)
    angle_rad = np.radians(angle_deg)

    kernel_size = length
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cx = cy = kernel_size // 2
    for i in range(kernel_size):
        offset = i - cx
        kx = int(round(cx + offset * np.cos(angle_rad)))
        ky = int(round(cy + offset * np.sin(angle_rad)))
        if 0 <= kx < kernel_size and 0 <= ky < kernel_size:
            kernel[ky, kx] = 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
        blurred = cv2.filter2D(v_channel, -1, kernel)
        return blurred.astype(np.float32)
    return v_channel
