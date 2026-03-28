# Runtime & Environment Contract

## Execution Environment

Assume:
- Google Colab
- Python 3.10
- No private network access

## Allowed Libraries

Prefer:
- Python standard library
- numpy
- random
- math
- matplotlib
- TensorFlow/Keras

Avoid:
- PyTorch
- Databricks
- Closed source packages
- Cutting-edge Python features

## Reproducibility Rules

- All randomness must be seeded
- All data must be generated inside the notebook
- Notebook must run top-to-bottom without edits

## Dependency Handling

If installing packages:
- Use pip
- Keep installs minimal
- First cell must confirm imports succeed