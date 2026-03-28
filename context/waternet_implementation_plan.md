# WaterNet v2 — Implementation Plan for Claude Code

## Project Overview

Altitude estimation system for drones flying over water surfaces, based on CNN regression over the Value channel (HSV) of captured images. The current model uses a custom sequential CNN trained on ~188,560 synthetic images (Blender) across 21 altitude levels (50–800 cm), achieving RMSE ≈ 26.4 cm in the synthetic domain, with significant systematic bias during inference on real-world images.

This document consolidates the recommendations from two analysis sessions:

- **Session 1** — Analysis of the existing algorithm: identification of architectural problems, HSV inconsistency, prediction bias, and advanced feature engineering proposals.
- **Session 2** — State-of-the-art research: architecture benchmarks (ResNet50 as reference), academic evaluation metrics, transfer learning strategies for single-channel input, and synthetic→real domain adaptation.

---

## Phase 0 — Critical Pipeline Corrections

> **Priority: BLOCKING** — Without these corrections, the rest of the plan has no experimental validity.

### 0.1 Fix the HSV vs Grayscale Inconsistency

**Identified problem:** The notebook uses `color_mode='grayscale'` in `ImageDataGenerator`, which applies the standard ITU-R BT.601 luminance conversion (`0.299R + 0.587G + 0.114B`). This is **not** equivalent to the HSV Value channel, which is defined as `max(R, G, B)`. The entire premise of the work (isolating maxima/minima via the V channel) is compromised.

```python
# WRONG (current): standard grayscale via ImageDataGenerator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, color_mode='grayscale'
)

# CORRECT: explicitly extract the HSV Value channel
def loadAndExtractValueChannel(image_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Loads an RGB image, converts to HSV, and returns the normalized V channel."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]  # V channel = max(R, G, B)
    return value_channel.astype(np.float32) / 255.0
```

**Claude Code actions:**
1. Create a custom `tf.keras.utils.Sequence` that loads images in RGB, converts to HSV, and returns only the V channel.
2. Completely replace `ImageDataGenerator` with `flow_from_dataframe`.
3. Verify with histograms that the V channel pixel distribution is distinct from the grayscale conversion.

### 0.2 Target Normalization (Regression)

**Problem:** Targets are normalized by dividing by 800 (maximum distance), resulting in the range [0.0625, 1.0]. The distribution is not zero-centered, which impairs convergence with a linear output activation.

```python
# CORRECT: z-score normalization with training set statistics
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_targets = scaler.fit_transform(df_train[['distancia']].values)
# Save scaler for inverse transform at inference
# pred_real = scaler.inverse_transform(pred) * 800
```

**Claude Code actions:**
1. Implement `StandardScaler` on training targets.
2. Save `scaler.mean_` and `scaler.scale_` alongside the model (pickle or JSON).
3. Apply inverse transform at all evaluation points.

### 0.3 Proper Train/Validation/Test Split

**Problem:** The current split is 80/20 (train/test) with no separate validation set. Validation is done via `validation_split=0.33` inside the training `ImageDataGenerator`, which is functional but does not allow fine-grained control over stratification.

```python
# Stratified split by distance range
from sklearn.model_selection import train_test_split

df_train_val, df_test = train_test_split(
    df, test_size=0.15, random_state=42,
    stratify=pd.cut(df['distancia'], bins=21)
)
df_train, df_val = train_test_split(
    df_train_val, test_size=0.176, random_state=42,  # 0.176 * 0.85 ≈ 0.15
    stratify=pd.cut(df_train_val['distancia'], bins=21)
)
# Result: 70% train, 15% validation, 15% test
```

---

## Phase 1 — Architecture: Transfer Learning with ResNet50

> **Priority: HIGH** — The literature demonstrates decisive superiority over custom CNNs for this type of task.

### 1.1 Baseline: ResNet50 with Channel Replication

```python
def buildResnet50AltitudeModel(input_shape: tuple[int, int, int] = (224, 224, 1)) -> tf.keras.Model:
    """Builds a regression model with ResNet50 backbone for single-channel input."""
    # V channel input
    input_layer = tf.keras.Input(shape=input_shape)

    # Replicate single channel to 3 channels (compatible with ImageNet weights)
    x = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])

    # ImageNet normalization
    x = tf.keras.applications.resnet50.preprocess_input(x)

    # Initially frozen backbone
    backbone = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )
    backbone.trainable = False
    x = backbone(x)

    # Regression head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)

    return tf.keras.Model(inputs=input_layer, outputs=output)
```

### 1.2 Progressive Fine-Tuning Protocol

```
Stage 1: Train only the head (frozen backbone)
  - Epochs: 10-15
  - LR: 1e-3 (AdamW, weight_decay=0.01)
  - Goal: converge the regression head

Stage 2: Unfreeze last 2 residual blocks (conv5_x)
  - Epochs: 15-20
  - LR backbone: 1e-5  |  LR head: 1e-4
  - BatchNorm: training=False (preserve running stats)

Stage 3 (optional): Unfreeze conv4_x
  - Epochs: 10
  - LR backbone: 1e-6  |  LR head: 5e-5
  - Monitor overfitting via val_loss
```

### 1.3 Architectures for Comparison (Ablation Study)

| Model               | Parameters  | Justification                          |
|----------------------|-------------|----------------------------------------|
| Custom CNN (current) | ~1.2M       | Existing work baseline                 |
| ResNet50             | ~25.6M      | Best result in the literature          |
| EfficientNetV2-B0    | ~5.3M       | Efficiency/accuracy trade-off          |
| MobileNetV3-Small    | ~2.5M       | Edge deployment feasibility (drone)    |

---

## Phase 2 — Advanced Feature Engineering

> **Priority: MEDIUM-HIGH** — Complements the CNN with physical domain knowledge.

### 2.1 Frequency Domain Analysis (FFT)

Water texture changes with altitude: at low altitude, high-frequency ripples dominate; at high altitude, the surface becomes homogeneous (low frequency). The 2D FFT captures this transition.

```python
def computeFrequencyFeatures(image: np.ndarray) -> np.ndarray:
    """Extracts spectral magnitude map via 2D FFT."""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    return (magnitude / magnitude.max()).astype(np.float32)
```

**Usage:** Create a multi-channel input `[V_channel, FFT_magnitude]` or as a separate branch in a multi-input architecture.

### 2.2 Gradient Detection (Edges)

Gradient maps (Sobel/Laplacian) encode texture intensity, which inversely correlates with altitude.

```python
def computeGradientMagnitude(image: np.ndarray) -> np.ndarray:
    """Computes gradient magnitude via Sobel filters."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return (magnitude / (magnitude.max() + 1e-8)).astype(np.float32)
```

### 2.3 Multi-Input Architecture with Feature Fusion

Combines an image branch (CNN) with a handcrafted features branch (FFT stats, gradients, histograms).

```python
def buildMultiInputModel() -> tf.keras.Model:
    """Model with visual and statistical feature fusion."""
    # Branch 1: Image (V channel) → ResNet50
    img_input = tf.keras.Input(shape=(224, 224, 1), name='image')
    x_img = tf.keras.layers.Concatenate()([img_input, img_input, img_input])
    backbone = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False
    )
    x_img = backbone(x_img)
    x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
    x_img = tf.keras.layers.Dense(128, activation='relu')(x_img)

    # Branch 2: Statistical features
    feat_input = tf.keras.Input(shape=(12,), name='features')
    # [mean_v, std_v, skew_v, kurt_v, fft_energy_low, fft_energy_mid,
    #  fft_energy_high, grad_mean, grad_std, entropy, glcm_contrast, glcm_homogeneity]
    x_feat = tf.keras.layers.Dense(64, activation='relu')(feat_input)
    x_feat = tf.keras.layers.Dense(32, activation='relu')(x_feat)

    # Fusion
    merged = tf.keras.layers.Concatenate()([x_img, x_feat])
    x = tf.keras.layers.Dense(128, activation='relu')(merged)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)

    return tf.keras.Model(inputs=[img_input, feat_input], outputs=output)
```

---

## Phase 3 — Training and Optimization

> **Priority: HIGH** — Correct training configuration is as important as the architecture.

### 3.1 Compilation and Loss Function

```python
# Huber loss: combines MSE (near zero) with MAE (outliers)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=0.01, clipnorm=1.0
    ),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
    ]
)
```

### 3.2 Callbacks

```python
callbacks = [
    # EarlyStopping with best model restoration
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=12, restore_best_weights=True
    ),
    # Checkpoint by val_loss
    tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model_{epoch:02d}_{val_rmse:.4f}.keras',
        monitor='val_rmse', save_best_only=True
    ),
    # Learning rate scheduling: cosine annealing with warmup
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
    ),
    # TensorBoard for visualization
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
]
```

### 3.3 Data Augmentation for Water Surfaces

Nadir view over water is **rotationally invariant** — any rotation preserves the altitude label.

```python
def buildAugmentationPipeline() -> tf.keras.Sequential:
    """Augmentation pipeline optimized for water surface images."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(factor=1.0),  # Full 360°
        tf.keras.layers.RandomBrightness(factor=0.15),
        tf.keras.layers.RandomContrast(factor=0.15),
        tf.keras.layers.GaussianNoise(stddev=0.02),
    ])
```

Additional augmentation via OpenCV in the custom data loader:
- **Directional motion blur** (already present in the synthetic dataset, but with additional granularity).
- **Random gamma correction** to simulate solar lighting variation.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for contrast normalization.

---

## Phase 4 — Bias Calibration and Post-Processing

> **Priority: MEDIUM** — Required to compensate for systematic prediction bias.

### 4.1 Calibration via Spline Interpolation

The current model exhibits non-linear bias: it underestimates low altitudes and overestimates intermediate heights. Post-training calibration with cubic spline.

```python
from scipy.interpolate import UnivariateSpline

def calibratePredictions(
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray
) -> np.ndarray:
    """Calibrates predictions using a spline fitted on the validation set."""
    # Group by distance bins and compute mean
    bins = np.linspace(val_pred.min(), val_pred.max(), 30)
    bin_means_pred = []
    bin_means_true = []
    for i in range(len(bins) - 1):
        mask = (val_pred >= bins[i]) & (val_pred < bins[i + 1])
        if mask.sum() > 10:
            bin_means_pred.append(val_pred[mask].mean())
            bin_means_true.append(val_true[mask].mean())

    # Fit spline: maps pred → true
    spline = UnivariateSpline(bin_means_pred, bin_means_true, s=0.1)
    return spline(test_pred)
```

### 4.2 Physical Constraint Layer

Add a constraint on the last layer to ensure predictions remain within the valid range [50, 800] cm:

```python
class ClampedLinear(tf.keras.layers.Layer):
    """Linear activation with clamping for the physical altitude range."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(inputs, self.min_val, self.max_val)
```

---

## Phase 5 — Metrics and Academic Evaluation

> **Priority: HIGH** — Essential for publication in IEEE/ACM/Springer/Elsevier venues.

### 5.1 Full Metrics Suite

References: Eigen et al. (NIPS 2014) for depth estimation; Hodson (2022, GMD) for MAE vs RMSE selection.

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
import numpy as np

def computeFullMetrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes the full metrics suite for altitude regression."""
    errors = y_true - y_pred
    return {
        'MAE (cm)': mean_absolute_error(y_true, y_pred),
        'RMSE (cm)': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MedAE (cm)': median_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE (%)': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'Max Error (cm)': np.max(np.abs(errors)),
        'Error Std (cm)': np.std(errors),
        'AbsRel': np.mean(np.abs(errors) / y_true),
        'δ < 1.25': np.mean(np.maximum(y_pred / y_true, y_true / y_pred) < 1.25),
        'δ < 1.25²': np.mean(np.maximum(y_pred / y_true, y_true / y_pred) < 1.5625),
    }
```

### 5.2 Stratified Evaluation by Altitude Range

Essential for identifying where the model fails:

```python
altitude_bins = [50, 100, 200, 300, 400, 500, 600, 700, 800]
# For each pair of consecutive bins, compute metrics separately
# Present as a table and a box plot by range
```

### 5.3 Publication-Quality Visualizations (Seaborn)

Required plots for the paper:

1. **Training curves** — Loss and RMSE (train + validation) per epoch, with Seaborn academic theme.
2. **Scatter plot** — Predictions vs Ground Truth with identity line and linear regression, colored by range.
3. **Residual plot** — Error vs Ground Truth to visualize systematic bias.
4. **Box plot** — Error distribution per altitude range.
5. **Confusion matrix heatmap** — If discretizing into altitude classes.
6. **Architecture comparison** — Grouped bars of MAE/RMSE for each model in the ablation study.
7. **Error histogram** — Overall distribution with fitted Gaussian.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Standard academic configuration
sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})
```

---

## Phase 6 — Domain Adaptation (Synthetic → Real)

> **Priority: MEDIUM** — Required for real-world scenario validation (Paraíba river, Cabedelo).

### 6.1 Mixed Training Strategy

Reference: da Silva Neto et al. (WVC/SBC) demonstrated the viability of the Blender → Real pipeline for the same scenario; SeaDroneSim2 showed +15% improvement with only 10% real data.

```
1. Pre-train on the full synthetic dataset (~188k images)
2. Fine-tune with annotated real data (even with just a few hundred samples)
3. Apply domain randomization during pre-training:
   - Vary water color, turbidity, solar illumination
   - Add realistic sensor noise
   - Simulate atmospheric conditions (haze, sun glare)
```

### 6.2 CycleGAN for Visual Adaptation

Transform synthetic images into real visual style (or vice-versa) to reduce the domain gap.

### 6.3 Cross-Domain Evaluation

```
Separate metrics for:
- Synthetic Train / Synthetic Test (in-domain)
- Synthetic Train / Real Test (cross-domain, zero-shot)
- Synthetic Train + Real Fine-tune / Real Test (adapted)
```

---

## Phase 7 — Code Refactoring

> **Priority: ONGOING** — Follow Google Python Style Guide.

### 7.1 Project Structure

```
waternet_v2/
├── configs/
│   └── default.yaml           # Configurable hyperparameters
├── data/
│   ├── dataset.py             # Custom tf.keras.utils.Sequence
│   ├── augmentation.py        # Augmentation pipeline
│   └── preprocessing.py       # HSV extraction, FFT, gradients
├── models/
│   ├── resnet_baseline.py     # ResNet50 transfer learning
│   ├── custom_cnn.py          # Original CNN (baseline)
│   ├── multi_input.py         # Multi-input architecture
│   └── layers.py              # Custom layers (ClampedLinear)
├── training/
│   ├── train.py               # Main training loop
│   ├── callbacks.py           # Custom callbacks
│   └── schedulers.py          # Learning rate schedulers
├── evaluation/
│   ├── metrics.py             # Full metrics suite
│   ├── visualization.py       # Seaborn plots for publication
│   └── calibration.py         # Spline calibration
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_train.ipynb         # Interactive training
│   └── 03_evaluation.ipynb    # Evaluation and plots
└── requirements.txt
```

### 7.2 Code Standards

- **Type hints** on all functions.
- **Google style docstrings** on all public classes and functions.
- **Constants** in UPPER_SNAKE_CASE, configurable via YAML.
- **No hardcoded paths** — use `pathlib.Path` and configuration.
- **Reproducibility** — set all seeds (Python, NumPy, TF).

```python
def setGlobalSeed(seed: int = 42) -> None:
    """Sets global seeds for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
```

---

## Recommended Execution Order

| Step | Phase | Description                                       | Dependency |
|------|-------|---------------------------------------------------|------------|
| 1    | 7     | Create project structure and configs              | —          |
| 2    | 0     | Fix HSV pipeline, normalization, split            | Step 1     |
| 3    | 1.1   | Implement ResNet50 baseline (V channel replicated)| Step 2     |
| 4    | 3     | Configure training (AdamW, Huber, callbacks)      | Step 3     |
| 5    | 5     | Implement metrics and visualizations              | Step 4     |
| 6    | 1.2   | Progressive fine-tuning                           | Step 5     |
| 7    | 1.3   | Train comparative models (ablation)               | Step 6     |
| 8    | 2     | Feature engineering (FFT, gradients)              | Step 6     |
| 9    | 2.3   | Multi-input model with feature fusion             | Step 8     |
| 10   | 4     | Bias calibration (spline)                         | Step 7     |
| 11   | 6     | Domain adaptation and real data testing           | Step 10    |
| 12   | 5.3   | Generate final publication plots                  | Step 11    |
