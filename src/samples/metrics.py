import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PATH = 'samples/section1/'
CSV_PATH     = SAMPLES_PATH + 'all_readings.csv'
OUTPUT_PATH  = SAMPLES_PATH + 'output/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
PALETTE = sns.color_palette('tab10')

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['gt_fus'] = (df['height'] + df['altitude']) / 2

models = ['resnet50', 'resnet50_fus', 'waternet', 'waternet_fus_lt', 'lidar']
labels = ['ResNet50', 'RN50-Fus', 'WaterNet', 'WN-FusLT', 'LiDAR']
gt = df['gt_fus']

# ── Metrics ───────────────────────────────────────────────────────────────────
records = []
ratio = lambda y_pred: np.maximum(y_pred / (gt + 1e-8), gt / (y_pred + 1e-8))
for m, lbl in zip(models, labels):
    err = df[m] - gt
    records.append({
        'Model': lbl,
        'MAE (m)':   round(err.abs().mean(), 3),
        'RMSE (m)':  round(np.sqrt((err**2).mean()), 3),
        'Bias (m)':  round(err.mean(), 3),
        'R2':    round(1 - np.sum(err**2) / np.sum((gt - gt.mean())**2), 3),
        'MAPE (%)': round((err.abs() / gt.abs()).mean() * 100, 2),
        'MedianAE (m)': round(err.abs().median(), 3),
        'P95 Error (m)': round(np.percentile(err.abs(), 95), 3),
        'Max error (m)': round(err.abs().max(), 3),
        'Error std (m)': round(err.std(), 3)
    })
metrics_df = pd.DataFrame(records)
metrics_df.to_csv(OUTPUT_PATH + 'model_metrics.csv', index=False)
print(metrics_df.to_string(index=False))

while True:
    pass

# ── Chart 1: Grouped bar – MAE & RMSE ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
bar_df = metrics_df.melt(
    id_vars='Model', value_vars=['MAE', 'RMSE'],
    var_name='Metric', value_name='Error (m)')
sns.barplot(data=bar_df, x='Model', y='Error (m)', hue='Metric',
            palette=['#4C72B0', '#DD8452'], ax=ax1)
ax1.set_title('Model Error vs Fusion Sensor Ground Truth', fontsize=13, fontweight='bold')
ax1.set_xlabel('Model')
ax1.set_ylabel('Error (m)')
ax1.legend(title='', loc='upper right')
sns.despine()
fig1.tight_layout()
fig1.savefig(OUTPUT_PATH + 'benchmark_errors.png', dpi=150)
plt.close(fig1)

# ── Chart 2: Bias bar ─────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
colors = ['#d62728' if b > 0 else '#1f77b4' for b in metrics_df['Bias']]
sns.barplot(data=metrics_df, x='Model', y='Bias', palette=colors, ax=ax2)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_title('Systematic Bias per Model', fontsize=13, fontweight='bold')
ax2.set_xlabel('Model')
ax2.set_ylabel('Bias (m)')
# Annotate each bar with its value
for bar, val in zip(ax2.patches, metrics_df['Bias']):
    offset = 0.03 if val >= 0 else -0.07
    ax2.text(bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        f'{val:+.3f}', ha='center', va='bottom', fontsize=10)
sns.despine()
fig2.tight_layout()
fig2.savefig(OUTPUT_PATH + 'benchmark_bias.png', dpi=150)
plt.close(fig2)

# ── Chart 3: Scatter – Predicted vs GT ────────────────────────────────────────
df = df.sample(600, random_state=42).sort_values('gt_fus')
fig3, ax3 = plt.subplots(figsize=(8, 7))
scatter_palette = sns.color_palette('tab10', n_colors=len(models))
for (m, lbl), color in zip(zip(models, labels), scatter_palette):
    ax3.scatter(df['gt_fus'], df[m], label=lbl,
                alpha=0.45, s=18, color=color)
line_range = [df['gt_fus'].min(), df['gt_fus'].max()]
ax3.plot(line_range, line_range, 'k--', linewidth=1.5, label='Ideal')
ax3.set_title('Predicted vs Ground Truth Altitude', fontsize=13, fontweight='bold')
ax3.set_xlabel('GT (m)')
ax3.set_ylabel('Predicted (m)')
ax3.xaxis.set_major_locator(ticker.MaxNLocator(10))
ax3.yaxis.set_major_locator(ticker.MaxNLocator(10))
ax3.legend(loc='upper right', framealpha=0.85)
sns.despine()
fig3.tight_layout()
fig3.savefig(OUTPUT_PATH + 'benchmark_scatter.png', dpi=150)
plt.close(fig3)

print("Done — charts saved to", OUTPUT_PATH)