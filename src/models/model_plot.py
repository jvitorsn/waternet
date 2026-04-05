from keras.models import Model, load_model
from keras.utils import plot_model

"""Module for loading Keras models and visualizing their architecture diagrams.
Example usage:
    model = load_model('path/to/model.h5')
    plot_model_architecture(model, output_path='model_diagram.png')
"""

def load_keras_model(model_path: str) -> Model:
    """Loads a Keras model from the specified file path.

    Args:
        model_path: Path to the saved Keras model file (.h5 or .keras).

    Returns:
        A compiled Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file format is not supported.
    """
    return load_model(model_path)

MODELS_PATH = "models/"
RESNET50 = MODELS_PATH + 'resnet50/resnet50_final.keras'
RESNET50_FUSION = MODELS_PATH + 'resnet50_fusion/resnet50_fusion_final.keras'
WATERNET_V2 = MODELS_PATH + 'waternet_v2/waternet_v2.keras'
WATERNET_V2_FUSION = MODELS_PATH + 'waternet_v2_fusion/waternet_v2_fusion_lite.keras'

model_paths = [RESNET50, RESNET50_FUSION, WATERNET_V2, WATERNET_V2_FUSION]

for model_path in model_paths:
    model = load_keras_model(f"{model_path}")
    plot_model(
        model=model,
        to_file=f"{model_path.split('/')[-1].split('.')[0]}.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="LR",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
        splines="ortho"
    )