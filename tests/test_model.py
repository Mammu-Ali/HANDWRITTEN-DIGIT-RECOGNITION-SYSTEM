"""
Unit tests for the trained CNN digit recognition model.
"""

import os
import numpy as np
import pytest
from tensorflow import keras

MODEL_PATH = "digit_model.keras"


@pytest.fixture(scope="session")
def model():
    """Load the trained Keras model for inference testing."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file {MODEL_PATH} not found. Run train_model.py first.")
    return keras.models.load_model(MODEL_PATH)


def test_model_file_exists():
    """Verify that the model file was generated and is not empty."""
    assert os.path.exists(MODEL_PATH), f"Model file {MODEL_PATH} does not exist."
    assert os.path.getsize(MODEL_PATH) > 100_000, "Model file size is too small."


def test_model_input_output_shapes(model):
    """Verify expected CNN input and output shapes."""
    # Input shape: (None, 28, 28, 1)
    input_shape = model.input_shape
    assert input_shape == (None, 28, 28, 1)

    # Output shape: (None, 10)
    output_shape = model.output_shape
    assert output_shape == (None, 10)


def test_model_predict_probabilities(model):
    """Verify inference returns a valid 10-class probability distribution."""
    # Synthetic blank / test image of shape (1, 28, 28, 1)
    sample_tensor = np.zeros((1, 28, 28, 1), dtype="float32")
    probabilities = model.predict(sample_tensor, verbose=0)

    assert probabilities.shape == (1, 10)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-3)
