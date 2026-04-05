"""Custom data loader and dataset-split utilities for WaterNet v2.

Replaces the original ``ImageDataGenerator(color_mode='grayscale')`` with a
``tf.keras.utils.Sequence`` that explicitly extracts the HSV Value channel
(Phase 0.1 of the implementation plan) and applies the stratified 70/15/15
split strategy (Phase 0.3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from waternet_v2.data.augmentation import WaterAugmenter
from waternet_v2.data.preprocessing import (
    extract_feature_vector,
    load_and_extract_value_channel,
)


# --------------------------------------------------------------------------- #
# Dataset split
# --------------------------------------------------------------------------- #

def make_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.176,
    n_bins: int = 21,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce a stratified 70 / 15 / 15 train/val/test split.

    Stratification is performed on altitude bins so that each split
    contains proportional representation of every altitude level.

    Args:
        df: DataFrame with at least a ``distancia`` (altitude in cm) column.
        test_size: Fraction of total data reserved for the test set.
        val_size: Fraction of train+val data reserved for validation.
            The default (0.176) yields ≈ 15% of the full dataset.
        n_bins: Number of equal-width altitude bins used for stratification.
        seed: Random seed.

    Returns:
        (df_train, df_val, df_test)
    """
    strata = pd.cut(df["distancia"], bins=n_bins, labels=False)

    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )

    strata_tv = pd.cut(df_train_val["distancia"], bins=n_bins, labels=False)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size,
        random_state=seed,
        stratify=strata_tv,
    )

    print(
        f"[Split] train={len(df_train):,}  val={len(df_val):,}  "
        f"test={len(df_test):,}  total={len(df):,}"
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Custom Sequence loader
# --------------------------------------------------------------------------- #

class WaterDataSequence(tf.keras.utils.Sequence):
    """Batch generator producing (image, features) → altitude.

    Each sample goes through:
    1. Load RGB image from disk.
    2. Resize to ``target_size``.
    3. Extract HSV Value channel (V = max(R,G,B)).
    4. Optionally augment (training only).
    5. Extract 12-element handcrafted feature vector.
    6. Return ``{"image_input": V[..., np.newaxis], "feature_input": feat}``
       and normalised altitude (targets already scaled by the caller's scaler).

    Args:
        dataframe: DataFrame with columns ``nome`` and ``distancia``.
        image_dir: Directory containing the image files.
        targets: Pre-scaled target array (same length as dataframe).
        target_size: (width, height) to resize images to.
        batch_size: Number of samples per batch.
        augmenter: Optional ``WaterAugmenter`` instance; only applied when
            ``training=True``.
        training: Whether to apply augmentation and shuffle.
        shuffle: Whether to shuffle indices at the end of each epoch.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str | Path,
        targets: np.ndarray,
        target_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        augmenter: WaterAugmenter | None = None,
        training: bool = True,
        shuffle: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.targets = targets
        self.target_size = target_size
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.training = training
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    # ── Sequence interface ─────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, batch_idx: int) -> tuple[dict, np.ndarray]:
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.df))
        indices = self.indices[start:end]

        batch_imgs: list[np.ndarray] = []
        batch_feats: list[np.ndarray] = []

        for idx in indices:
            img_path = self.image_dir / self.df.loc[idx, "nome"]
            v = load_and_extract_value_channel(str(img_path), self.target_size)

            if self.augmenter is not None:
                v = self.augmenter.augment(v, training=self.training)

            feat = extract_feature_vector(v)

            batch_imgs.append(v[..., np.newaxis])   # (H, W, 1)
            batch_feats.append(feat)

        X = {
            "image_input": np.array(batch_imgs, dtype=np.float32),
            "feature_input": np.array(batch_feats, dtype=np.float32),
        }
        y = self.targets[indices].astype(np.float32)
        return X, y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)
