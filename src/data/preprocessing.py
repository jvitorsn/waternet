"""Image preprocessing and handcrafted feature extraction.

This module implements three core feature-engineering algorithms from the paper:

1. **HSV Value Channel Isolation** (Section 2.3.1)
   V = max(R, G, B) — preserves specular reflection peaks unlike grayscale
   (Y = 0.299R + 0.587G + 0.114B attenuates the brightness maxima that
   encode altitude information).

2. **2D FFT Frequency Analysis** (Section 2.3.2)
   At low altitude the camera resolves individual capillary waves → high
   spatial frequency energy.  As altitude grows, the surface averages out →
   energy shifts toward low frequencies.

3. **Shi-Tomasi Good Features to Track** (Section 2.3.3)
   Corner density (= sun-glint hotspot density) decreases with altitude
   because fewer individual wave facets produce detectable specular peaks.

The 12-element feature vector is:
    [mean_v, std_v, skew_v, kurt_v,
     fft_energy_low, fft_energy_mid, fft_energy_high,
     grad_mean, grad_std, entropy,
     shi_tomasi_count, local_std_mean]
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# HSV Value Channel
# --------------------------------------------------------------------------- #

def extract_value_channel(
    img_rgb: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Load an RGB image, resize, and return the normalised HSV Value channel.

    The Value channel is defined as V = max(R, G, B) (Eq. 2.11 of the paper).
    This differs from standard grayscale conversion (ITU-R BT.601 luminance:
    Y = 0.299R + 0.587G + 0.114B), which attenuates the specular intensity
    peaks that encode altitude-related texture information.

    Args:
        img_rgb: Input image array of shape (H, W, 3) in RGB order.
        target_size: (width, height) to resize to before channel extraction.

    Returns:
        Normalised Value channel, float32 in [0, 1], shape (*target_size,).
    """
    if img_rgb.shape[:2] != (target_size[1], target_size[0]):
        img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
    return v_channel


def load_and_extract_value_channel(
    image_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Read image from disk (BGR), resize, convert to HSV, return V channel.

    Args:
        image_path: Absolute or relative path to an image file.
        target_size: (width, height) to resize to.

    Returns:
        Normalised V channel, float32 in [0, 1].

    Raises:
        FileNotFoundError: If the image cannot be loaded by OpenCV.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
    return v_channel


def grayscale_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Apply ITU-R BT.601 luminance conversion (for comparison with V channel).

    Y = 0.299R + 0.587G + 0.114B

    Args:
        img_rgb: Input image, shape (H, W, 3), uint8 or float.

    Returns:
        Grayscale image, float32 in [0, 1].
    """
    img = img_rgb.astype(np.float32) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2D FFT Frequency Features
# --------------------------------------------------------------------------- #

def compute_fft_magnitude(v_channel: np.ndarray) -> np.ndarray:
    """Compute the 2D FFT magnitude spectrum (log-scaled, normalised).

    The 2D DFT is defined as (Eq. 2.12):
        F(u,v) = Σ_x Σ_y f(x,y) · exp(-j2π(ux/M + vy/N))

    Args:
        v_channel: Single-channel image, float32 in [0, 1].

    Returns:
        Log-scaled magnitude spectrum, float32 in [0, 1], same shape.
    """
    f_transform = np.fft.fft2(v_channel)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift)).astype(np.float32)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude /= max_val
    return magnitude


def compute_fft_energy_bands(
    v_channel: np.ndarray,
    low_cutoff: float = 0.2,
    mid_cutoff: float = 0.6,
) -> tuple[float, float, float]:
    """Partition FFT energy into low / mid / high radial frequency bands.

    Radial normalised distance r ∈ [0, 1] from DC component:
    - low:  r < low_cutoff   → large-scale waves / smooth surface
    - mid:  low ≤ r < mid    → medium texture
    - high: r ≥ mid_cutoff   → fine ripples / capillary waves

    Physical basis: at low altitude high-frequency energy dominates; at high
    altitude energy concentrates near DC (low-pass averaging effect).

    Args:
        v_channel: Single-channel float32 image in [0, 1].
        low_cutoff: Normalised radial threshold for low-frequency band.
        mid_cutoff: Normalised radial threshold for mid-frequency band.

    Returns:
        (energy_low, energy_mid, energy_high) — each a fraction of total.
    """
    h, w = v_channel.shape
    F = np.fft.fftshift(np.fft.fft2(v_channel))
    mag = np.abs(F)

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r_norm = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (
        np.sqrt(cy ** 2 + cx ** 2) + 1e-10
    )

    total = mag.sum() + 1e-10
    e_low  = float(mag[r_norm < low_cutoff].sum() / total)
    e_mid  = float(mag[(r_norm >= low_cutoff) & (r_norm < mid_cutoff)].sum() / total)
    e_high = float(mag[r_norm >= mid_cutoff].sum() / total)
    return e_low, e_mid, e_high


# --------------------------------------------------------------------------- #
# Gradient Features (Sobel)
# --------------------------------------------------------------------------- #

def compute_gradient_magnitude(v_channel: np.ndarray) -> np.ndarray:
    """Compute pixel-wise Sobel gradient magnitude.

    Gradient maps encode texture intensity which inversely correlates with
    altitude (denser edges at low altitude due to resolved wave structure).

    Args:
        v_channel: Single-channel float32 image in [0, 1].

    Returns:
        Normalised gradient magnitude, float32 in [0, 1], same shape.
    """
    gx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    max_val = magnitude.max() + 1e-8
    return magnitude / max_val


# --------------------------------------------------------------------------- #
# Shi-Tomasi Corner Detection
# --------------------------------------------------------------------------- #

def count_shi_tomasi_features(
    v_channel: np.ndarray,
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: float = 5.0,
) -> int:
    """Count Shi-Tomasi corners (Eq. 2.15) as a proxy for glint density.

    The Shi-Tomasi criterion: min(λ₁, λ₂) > τ  (Eq. 2.16)
    where λ₁, λ₂ are eigenvalues of the local structure tensor M.

    Specular reflection peaks create high-gradient regions detected as
    corners. Feature count DECREASES with altitude because fewer individual
    wave facets produce resolvable reflections.

    Args:
        v_channel: Single-channel float32 image in [0, 1].
        max_corners: Maximum corners to return.
        quality_level: Minimum quality of corners (fraction of max).
        min_distance: Minimum Euclidean distance between corners.

    Returns:
        Number of detected corners.
    """
    img_u8 = (v_channel * 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        img_u8,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )
    return int(len(corners)) if corners is not None else 0


# --------------------------------------------------------------------------- #
# Full 12-Element Feature Vector
# --------------------------------------------------------------------------- #

FEATURE_NAMES = [
    "mean_v",
    "std_v",
    "skew_v",
    "kurt_v",
    "fft_energy_low",
    "fft_energy_mid",
    "fft_energy_high",
    "grad_mean",
    "grad_std",
    "entropy",
    "shi_tomasi_count",
    "local_std_mean",
]


def extract_feature_vector(v_channel: np.ndarray) -> np.ndarray:
    """Extract the 12-element handcrafted feature vector from a V channel.

    Feature vector layout (matches FEATURE_NAMES):
    ┌─────────────────────┬─────────────────────────────────────────────────┐
    │ Indices 0–3         │ Pixel intensity statistics                      │
    │ Indices 4–6         │ Radial FFT energy bands (low / mid / high)      │
    │ Indices 7–8         │ Sobel gradient statistics                       │
    │ Index  9            │ Normalised image entropy                        │
    │ Index  10           │ Shi-Tomasi corner count                         │
    │ Index  11           │ Mean of local 8×8 block standard deviations     │
    └─────────────────────┴─────────────────────────────────────────────────┘

    Args:
        v_channel: Single-channel float32 image in [0, 1].

    Returns:
        Feature vector, float32 array of length 12.
    """
    flat = v_channel.ravel()

    # ── 1-4: Pixel statistics ────────────────────────────────────────────────
    mean_v = float(flat.mean())
    std_v  = float(flat.std())
    skew_v = float(stats.skew(flat))
    kurt_v = float(stats.kurtosis(flat))

    # ── 5-7: FFT energy bands ────────────────────────────────────────────────
    e_low, e_mid, e_high = compute_fft_energy_bands(v_channel)

    # ── 8-9: Gradient statistics ─────────────────────────────────────────────
    gx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mean = float(grad_mag.mean())
    grad_std  = float(grad_mag.std())

    # ── 10: Image entropy ────────────────────────────────────────────────────
    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0), density=True)
    hist = hist + 1e-10
    entropy = float(-np.sum(hist * np.log2(hist)) / np.log2(32))

    # ── 11: Shi-Tomasi count ─────────────────────────────────────────────────
    shi_count = float(count_shi_tomasi_features(v_channel))

    # ── 12: Local texture (mean of 8×8 block stds) ───────────────────────────
    h, w = v_channel.shape
    patch = 8
    local_stds = [
        v_channel[i : i + patch, j : j + patch].std()
        for i in range(0, h - patch, patch)
        for j in range(0, w - patch, patch)
    ]
    local_std_mean = float(np.mean(local_stds)) if local_stds else 0.0

    return np.array(
        [mean_v, std_v, skew_v, kurt_v,
         e_low, e_mid, e_high,
         grad_mean, grad_std, entropy,
         shi_count, local_std_mean],
        dtype=np.float32,
    )
