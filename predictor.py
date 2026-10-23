"""
Inference module for Handwritten Digit Recognition.
Loads the trained Keras model, preprocesses the input image,
and returns digit classification predictions with confidence scores.
"""

import os
from typing import Any, Dict, Optional, Union
import numpy as np
from tensorflow import keras

from image_processing import preprocess_image

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.keras")


class DigitPredictor:
    """Class to load and manage model inference for digit recognition."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model: Optional[keras.Model] = None

    @property
    def model(self) -> keras.Model:
        """Lazily load the trained Keras model."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file '{self.model_path}' not found. Run train_model.py to train and export the model."
                )
            self._model = keras.models.load_model(self.model_path)
        return self._model

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Preprocess input image, run model inference, and return structured prediction results.

        Returns:
            dict containing:
                - digit (int): Predicted digit (0-9).
                - confidence (float): Probability score of top predicted digit.
                - probabilities (dict): Mapping of digit ('0'-'9') to float probability.
        """
        # Preprocess input image to tensor (1, 28, 28, 1) and retrieve visualization metadata
        tensor, preprocessing_meta = preprocess_image(image_input, return_metadata=True)

        # Run model inference
        raw_preds = self.model.predict(tensor, verbose=0)
        probabilities = raw_preds[0]

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])

        prob_distribution = {
            str(digit): round(float(prob), 4)
            for digit, prob in enumerate(probabilities)
        }

        # Identify top-3 highest probability classes sorted descending
        sorted_ranks = sorted(
            [{"digit": int(d), "probability": round(float(p), 4)} for d, p in enumerate(probabilities)],
            key=lambda item: item["probability"],
            reverse=True,
        )
        top_three = sorted_ranks[:3]

        return {
            "digit": predicted_digit,
            "confidence": round(confidence, 4),
            "probabilities": prob_distribution,
            "top_three": top_three,
            "preprocessing": preprocessing_meta,
        }


# Global default predictor instance for convenience
_default_predictor: Optional[DigitPredictor] = None


def get_default_predictor(model_path: str = DEFAULT_MODEL_PATH) -> DigitPredictor:
    """Retrieve or initialize the global default predictor instance."""
    global _default_predictor
    if _default_predictor is None or _default_predictor.model_path != model_path:
        _default_predictor = DigitPredictor(model_path=model_path)
    return _default_predictor


def predict_digit(image_input: Any, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Convenience function to predict digit from an image input."""
    predictor = get_default_predictor(model_path=model_path)
    return predictor.predict(image_input)
