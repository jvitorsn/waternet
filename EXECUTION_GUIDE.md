# WaterNet v2 — Execution Guide

This guide explains the `waternet_v2/` package structure, where to find key
insights, and how to run each phase of the implementation plan.

---

## Project Structure

```
waternet_v2/
├── configs/
│   ├── __init__.py          load_config() — deep-merge YAML with defaults
│   └── default.yaml         all hyperparameters (never hardcoded)
├── data/
│   ├── preprocessing.py     HSV V channel, FFT, Shi-Tomasi, feature vector
│   ├── augmentation.py      Keras pipeline + NumPy/OpenCV augmenter
│   └── dataset.py           WaterDataSequence + stratified split
├── models/
│   ├── layers.py            ClampedLinear (physical altitude constraint)
│   ├── custom_cnn.py        VGG-inspired lightweight CNN (baseline)
│   ├── resnet_baseline.py   ResNet50 transfer learning + fine-tuning helpers
│   └── multi_input.py       Late-fusion multi-input (CNN + features)
├── training/
│   ├── callbacks.py         EarlyStopping, Checkpoint, ReduceLR, TensorBoard
│   └── train.py             run_training_pipeline() — full end-to-end
├── evaluation/
│   ├── metrics.py           compute_full_metrics(), stratified evaluation
│   ├── calibration.py       SplineCalibrator (bias post-correction)
│   ├── visualization.py     7 publication-quality plots
│   └── gradcam.py           Grad-CAM for CNN explainability
└── requirements.txt
```

---

## Where Insights Appear

### Phase 0 — HSV vs Grayscale (Critical Fix)

**File:** `waternet_v2/data/preprocessing.py`
**Functions:** `extract_value_channel`, `grayscale_from_rgb`

The most important insight in the entire pipeline: `V = max(R,G,B)` retains
specular reflection peaks that ITU-R grayscale (`0.299R+0.587G+0.114B`) attenuates.

**How to see it:**
```python
from waternet_v2.data.preprocessing import extract_value_channel, grayscale_from_rgb
import cv2, matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("samples/synth_samples/100_diag_0005.jpg"), cv2.COLOR_BGR2RGB)
v   = extract_value_channel(img)
gs  = grayscale_from_rgb(img)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img);         axes[0].set_title("RGB")
axes[1].imshow(v, cmap="gray");  axes[1].set_title("HSV V channel (this work)")
axes[2].imshow(gs, cmap="gray"); axes[2].set_title("Grayscale (BT.601)")
plt.show()
# Key observation: bright glint spots are more prominent in the V channel.
```

### Phase 2 — FFT Frequency as Altitude Proxy

**File:** `waternet_v2/data/preprocessing.py`
**Functions:** `compute_fft_magnitude`, `compute_fft_energy_bands`
**Visualisation:** `waternet_v2/evaluation/visualization.py → plot_fft_altitude_relationship`

At low altitude (50 cm) → energy concentrates at **high radial frequencies**.
At high altitude (800 cm) → energy concentrates **near DC** (low frequencies).

```python
from waternet_v2.data.preprocessing import (
    load_and_extract_value_channel, compute_fft_energy_bands
)
import glob

for alt in [50, 200, 800]:
    paths = glob.glob(f"samples/synth_samples/{alt}_*.jpg")[:5]
    for p in paths:
        v = load_and_extract_value_channel(p)
        low, mid, high = compute_fft_energy_bands(v)
        print(f"alt={alt:3d} cm  |  low={low:.3f}  mid={mid:.3f}  high={high:.3f}")
# Expected: high-freq energy DECREASES as altitude INCREASES.
```

### Phase 2 — Shi-Tomasi Corner Density

**File:** `waternet_v2/data/preprocessing.py`
**Function:** `count_shi_tomasi_features`

Corner count is a proxy for specular glint density.  It should **decrease
monotonically** with altitude across altitude bins.

```python
from waternet_v2.data.preprocessing import (
    load_and_extract_value_channel, count_shi_tomasi_features
)
import glob, numpy as np

for alt in [50, 100, 200, 400, 800]:
    counts = [
        count_shi_tomasi_features(load_and_extract_value_channel(p))
        for p in glob.glob(f"samples/synth_samples/{alt}_*.jpg")[:20]
    ]
    print(f"alt={alt:3d} cm  |  corners={np.mean(counts):.1f} ± {np.std(counts):.1f}")
```

### Phase 1 — ResNet50 Baseline

**File:** `waternet_v2/models/resnet_baseline.py`
**Function:** `build_resnet50_altitude_model`

```python
from waternet_v2.models.resnet_baseline import build_resnet50_altitude_model
model = build_resnet50_altitude_model(input_shape=(224, 224, 1), freeze_backbone=True)
model.summary()
# ~25.6 M params total; ~1.0 M trainable in Stage 1 (head only).
```

### Phase 1 — Progressive Fine-Tuning

**File:** `waternet_v2/models/resnet_baseline.py`
**Function:** `unfreeze_top_blocks`

```python
from waternet_v2.models.resnet_baseline import unfreeze_top_blocks
# After Stage 1 converges, unfreeze conv5_x:
unfreeze_top_blocks(model, n_blocks=1)
# Recompile with lower LR and continue training.
```

### Phase 3 — Full Training Pipeline

**File:** `waternet_v2/training/train.py`
**Function:** `run_training_pipeline`

```python
from waternet_v2.training.train import run_training_pipeline

results = run_training_pipeline(
    model_type="multi_input",   # "custom_cnn" | "resnet50" | "multi_input"
    output_dir="outputs/run_01",
)
model   = results["model"]
history = results["history"]
scaler  = results["scaler_y"]
```

Saved artefacts:
- `outputs/run_01/waternet_v2_final.keras`   — best model weights
- `outputs/run_01/scaler_y.json`             — target scaler parameters
- `outputs/run_01/training_history.json`     — epoch-by-epoch metrics

### Phase 4 — Spline Bias Calibration

**File:** `waternet_v2/evaluation/calibration.py`
**Class:** `SplineCalibrator`

```python
import numpy as np
from waternet_v2.evaluation.calibration import SplineCalibrator

# After evaluating on the validation set:
calibrator = SplineCalibrator()
calibrator.fit(val_pred_cm, val_true_cm)
test_pred_calibrated = calibrator.transform(test_pred_cm)

# Save / reload
calibrator.save("outputs/calibration.json")
cal2 = SplineCalibrator.load("outputs/calibration.json")
```

**Key insight:** plot `val_pred` vs `val_true` before and after calibration —
the before scatter should show an S-curve bias; after calibration it should
align with the identity line.

### Phase 5 — Full Metrics Suite

**File:** `waternet_v2/evaluation/metrics.py`
**Functions:** `compute_full_metrics`, `print_metrics`, `evaluate_by_altitude_range`

```python
from waternet_v2.evaluation.metrics import (
    compute_full_metrics, print_metrics, evaluate_by_altitude_range, build_comparison_table
)

m = compute_full_metrics(y_true_cm, y_pred_cm)
print_metrics(m, title="WaterNet Multi-Input — Test Set")

strat = evaluate_by_altitude_range(y_true_cm, y_pred_cm)
print(strat.to_string(index=False))
```

**Key insight:** look at the `MAPE (%)` per altitude range — errors at 50 cm
are disproportionately high in percentage terms even if the absolute error is
similar to higher altitudes.

### Phase 5 — Publication Visualisations

**File:** `waternet_v2/evaluation/visualization.py`

```python
from waternet_v2.evaluation.visualization import (
    plot_training_curves, plot_scatter_pred_vs_true,
    plot_residuals, plot_error_boxplot,
    plot_error_histogram, plot_model_comparison,
    plot_fft_altitude_relationship,
)
import matplotlib.pyplot as plt

fig = plot_scatter_pred_vs_true(y_true_cm, y_pred_cm)
fig.savefig("scatter.png", dpi=300, bbox_inches="tight")

fig2 = plot_residuals(y_true_cm, y_pred_cm)
fig2.savefig("residuals.png", dpi=300, bbox_inches="tight")

plt.show()
```

### Phase 5 — Grad-CAM

**File:** `waternet_v2/evaluation/gradcam.py`
**Function:** `compute_gradcam`

```python
from waternet_v2.evaluation.gradcam import compute_gradcam, find_last_conv_layer
from waternet_v2.evaluation.visualization import plot_gradcam_overlay

conv_name = find_last_conv_layer(model)
cam = compute_gradcam(model, v_channel, conv_name)

fig = plot_gradcam_overlay(v_channel, cam, true_alt=150, pred_alt=162)
plt.show()
# Expected: heatmap concentrates on wave texture / specular spots, NOT image borders.
```

---

## Quick-Start: Ablation Comparison

```python
from waternet_v2.evaluation.metrics import build_comparison_table
from waternet_v2.evaluation.visualization import plot_model_comparison
import matplotlib.pyplot as plt

comparison = build_comparison_table({
    "Mean Baseline":  (y_true, y_pred_mean),
    "Freq Heuristic": (y_true, y_pred_freq),
    "Custom CNN":     (y_true, y_pred_cnn),
    "Multi-Input":    (y_true, y_pred_multi),
    "Multi-Input+Cal":(y_true, y_pred_calibrated),
})
print(comparison.to_string(index=False))
fig = plot_model_comparison(comparison)
plt.show()
```

---

## Google Colab Setup

```python
# Mount Drive and install dependencies
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/waternet
!pip install -q -r waternet_v2/requirements.txt

# Verify
from waternet_v2.data.preprocessing import extract_feature_vector
import numpy as np
feat = extract_feature_vector(np.random.rand(224, 224).astype(np.float32))
print("Feature vector shape:", feat.shape)   # (12,)
print("Feature names: mean_v, std_v, skew_v, kurt_v,")
print("               fft_energy_low, fft_energy_mid, fft_energy_high,")
print("               grad_mean, grad_std, entropy,")
print("               shi_tomasi_count, local_std_mean")
```

---

## Configuration Override

All hyperparameters are in `waternet_v2/configs/default.yaml`.
To override for an experiment:

```yaml
# my_experiment.yaml
training:
  learning_rate: 5.0e-4
  batch_size: 64
  epochs: 150
data:
  image_size: [112, 112]
```

```python
from waternet_v2.training.train import run_training_pipeline
results = run_training_pipeline(
    model_type="resnet50",
    config_path="my_experiment.yaml",
    output_dir="outputs/experiment_resnet50",
)
```
