"""ResNet50 transfer-learning baseline (Phase 1 of the implementation plan).

The key adaptation for single-channel input is **channel replication**:
the V channel (H×W×1) is concatenated with itself three times to produce
a compatible (H×W×3) tensor that matches the ImageNet weight dimensions,
avoiding the need to randomly re-initialise the first conv layer.

Progressive fine-tuning protocol (Section 1.2):
    Stage 1 — Train only the regression head (frozen backbone).
    Stage 2 — Unfreeze the last two residual blocks (conv5_x).
    Stage 3 — Optionally unfreeze conv4_x for deeper adaptation.

Reference: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_resnet50_altitude_model(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    head_units: list[int] | None = None,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = True,
) -> keras.Model:
    """Build a ResNet50-based regression model for altitude estimation.

    The V channel is replicated to 3 channels then passed through
    ``resnet50.preprocess_input`` (zero-centres with ImageNet statistics)
    before entering the backbone.

    Args:
        input_shape: Model input shape; channel dim must be 1.
        head_units: Dense units in the regression head.
            Defaults to [256, 128].
        dropout_rate: Dropout applied between Dense layers.
        freeze_backbone: Whether to freeze all backbone layers initially.

    Returns:
        Uncompiled ``tf.keras.Model``.
    """
    if head_units is None:
        head_units = [256, 128]

    # ── Input & channel replication ──────────────────────────────────────── #
    inp = keras.Input(shape=input_shape, name="image_input")
    x = layers.Concatenate(name="channel_replicate")([inp, inp, inp])

    # ── ImageNet pre-processing ───────────────────────────────────────────── #
    x = tf.keras.applications.resnet50.preprocess_input(x)

    # ── Backbone ─────────────────────────────────────────────────────────── #
    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    backbone.trainable = not freeze_backbone
    x = backbone(x, training=False)    # BN layers run in inference mode

    # ── Regression head ───────────────────────────────────────────────────── #
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    for i, units in enumerate(head_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

    output = layers.Dense(1, activation="linear", name="output")(x)

    return keras.Model(inputs=inp, outputs=output, name="WaterNet_ResNet50")


# --------------------------------------------------------------------------- #
# Progressive fine-tuning helpers
# --------------------------------------------------------------------------- #

def unfreeze_top_blocks(
    model: keras.Model,
    n_blocks: int = 1,
    backbone_name: str = "resnet50",
) -> None:
    """Selectively unfreeze the top *n_blocks* residual blocks in-place.

    ResNet50 block names (from deepest):
        conv5_block3, conv5_block2, conv5_block1  → top block
        conv4_block6, ..., conv4_block1           → second block

    BatchNorm layers are kept in inference mode (``trainable=False``) to
    preserve running statistics accumulated on ImageNet.

    Args:
        model: The full model returned by ``build_resnet50_altitude_model``.
        n_blocks: Number of residual macro-blocks to unfreeze (1 = conv5_x).
        backbone_name: Name of the ResNet50 sub-model layer.
    """
    backbone: keras.Model = model.get_layer(backbone_name)
    backbone.trainable = True

    # Block prefixes in order from deepest
    block_prefixes = [f"conv{5 - i}" for i in range(n_blocks)]

    for layer in backbone.layers:
        is_in_target_block = any(layer.name.startswith(p) for p in block_prefixes)
        if not is_in_target_block or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    print(
        f"[Finetune] Unfrozen blocks: {block_prefixes}  |  "
        f"Trainable params: "
        f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}"
    )


def get_finetune_optimizer(
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 0.01,
) -> keras.optimizers.Optimizer:
    """Create an AdamW optimiser with per-layer learning-rate scaling.

    For simplicity this returns a single AdamW with ``lr_head``; in practice
    discriminative learning rates require a custom training loop or separate
    optimisers.  The returned optimiser uses the head LR, which should be
    applied when compiling the model for Stage 2 / Stage 3.

    Args:
        lr_backbone: Target learning rate for backbone parameters.
        lr_head: Target learning rate for regression head parameters.
        weight_decay: AdamW weight decay coefficient.

    Returns:
        Configured ``AdamW`` optimiser instance.
    """
    print(f"[Optimiser] backbone_lr={lr_backbone}  head_lr={lr_head}")
    return keras.optimizers.AdamW(
        learning_rate=lr_head,
        weight_decay=weight_decay,
        clipnorm=1.0,
    )
