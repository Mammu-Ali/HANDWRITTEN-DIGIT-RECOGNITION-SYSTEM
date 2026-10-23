"""
Unit tests for the image preprocessing module and prediction pipeline.
Tests image polarity handling, invalid inputs, blank detection, format filtering,
and output dimensions.
"""

import io
import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_processing import (
    BlankImageError,
    InvalidImageError,
    UnsupportedFormatError,
    preprocess_image,
)
from predictor import DigitPredictor, predict_digit


def _create_synthetic_digit_image(
    digit_char: str = "7",
    bg_color: int = 255,
    fg_color: int = 0,
    img_format: str = "PNG",
    size: tuple = (140, 140),
) -> bytes:
    """Helper to draw a simple high-contrast digit image and return its encoded bytes."""
    img = Image.new("L", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    if digit_char == "7":
        draw.line([(30, 30), (110, 30)], fill=fg_color, width=12)
        draw.line([(110, 30), (50, 110)], fill=fg_color, width=12)
    elif digit_char == "1":
        draw.line([(70, 30), (70, 110)], fill=fg_color, width=14)
    elif digit_char == "0":
        draw.ellipse([(35, 30), (105, 110)], outline=fg_color, width=12)
    else:
        draw.line([(30, 70), (110, 70)], fill=fg_color, width=12)

    buf = io.BytesIO()
    img.save(buf, format=img_format)
    return buf.getvalue()


class TestPreprocessing:
    """Tests for image_processing module."""

    def test_output_dimensions_and_type(self):
        """Verify preprocessed tensor shape is exactly (1, 28, 28, 1) with float32 type."""
        img_bytes = _create_synthetic_digit_image("7", bg_color=255, fg_color=0, img_format="PNG")
        tensor = preprocess_image(img_bytes)

        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (1, 28, 28, 1)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.min()
        assert tensor.max() <= 1.0

    def test_dark_digit_on_light_background(self):
        """
        Verify that a dark digit on a white background is inverted so that
        the background becomes 0.0 and the digit becomes > 0.0 (MNIST standard).
        """
        img_bytes = _create_synthetic_digit_image("1", bg_color=255, fg_color=0, img_format="PNG")
        tensor = preprocess_image(img_bytes)

        # Border corners should be 0.0 (black background in MNIST)
        assert tensor[0, 0, 0, 0] == 0.0
        assert tensor[0, 0, 27, 0] == 0.0
        assert tensor[0, 27, 0, 0] == 0.0
        assert tensor[0, 27, 27, 0] == 0.0
        # The center / digit region should have non-zero stroke pixels
        assert tensor[0, 14, 14, 0] > 0.1

    def test_light_digit_on_dark_background(self):
        """
        Verify that a white digit on a dark background is NOT inverted,
        maintaining the dark background as 0.0 and digit as > 0.0.
        """
        img_bytes = _create_synthetic_digit_image("1", bg_color=0, fg_color=255, img_format="PNG")
        tensor = preprocess_image(img_bytes)

        # Border corners should be 0.0
        assert tensor[0, 0, 0, 0] == 0.0
        assert tensor[0, 0, 27, 0] == 0.0
        assert tensor[0, 27, 0, 0] == 0.0
        assert tensor[0, 27, 27, 0] == 0.0
        # The center / digit region should have non-zero stroke pixels
        assert tensor[0, 14, 14, 0] > 0.1

    def test_accepts_jpeg_and_jpg_formats(self):
        """Verify JPEG format decoding and preprocessing."""
        jpeg_bytes = _create_synthetic_digit_image("7", bg_color=255, fg_color=0, img_format="JPEG")
        tensor = preprocess_image(jpeg_bytes)
        assert tensor.shape == (1, 28, 28, 1)

    def test_reject_unsupported_formats(self):
        """Verify that unsupported image formats (e.g. BMP, GIF) are rejected."""
        # Create a BMP image
        img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="BMP")

        with pytest.raises(UnsupportedFormatError) as exc_info:
            preprocess_image(buf.getvalue())
        assert "BMP" in str(exc_info.value)

    def test_reject_corrupt_or_empty_bytes(self):
        """Verify that corrupt or non-image bytes raise InvalidImageError."""
        with pytest.raises(InvalidImageError):
            preprocess_image(b"")

        with pytest.raises(InvalidImageError):
            preprocess_image(b"this is not a valid image byte stream")

    def test_reject_solid_white_blank_image(self):
        """Verify that a solid white image raises BlankImageError."""
        img = Image.new("L", (100, 100), color=255)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        with pytest.raises(BlankImageError):
            preprocess_image(buf.getvalue())

    def test_reject_solid_black_blank_image(self):
        """Verify that a solid black image raises BlankImageError."""
        img = Image.new("L", (100, 100), color=0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        with pytest.raises(BlankImageError):
            preprocess_image(buf.getvalue())

    def test_reject_negligible_noise(self):
        """Verify that an image with only tiny random noise pixels is rejected as blank."""
        img = Image.new("L", (100, 100), color=255)
        # Put 3 tiny stray pixels
        img.putpixel((10, 10), 0)
        img.putpixel((11, 10), 0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        with pytest.raises(BlankImageError):
            preprocess_image(buf.getvalue())


class TestPrediction:
    """Tests for predictor module and model inference."""

    def test_prediction_output_structure(self):
        """Verify that prediction returns the expected dictionary structure and values."""
        img_bytes = _create_synthetic_digit_image("7", bg_color=255, fg_color=0, img_format="PNG")
        result = predict_digit(img_bytes)

        assert isinstance(result, dict)
        assert "digit" in result
        assert "confidence" in result
        assert "probabilities" in result

        assert isinstance(result["digit"], int)
        assert 0 <= result["digit"] <= 9

        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

        probs = result["probabilities"]
        assert len(probs) == 10
        assert set(probs.keys()) == {str(i) for i in range(10)}
        assert pytest.approx(sum(probs.values()), rel=1e-2) == 1.0

    def test_prediction_recognizes_synthetic_seven(self):
        """Verify prediction accurately identifies synthetic digit 7."""
        img_bytes = _create_synthetic_digit_image("7", bg_color=255, fg_color=0, img_format="PNG")
        result = predict_digit(img_bytes)

        assert result["digit"] == 7
        assert result["confidence"] > 0.80

    def test_prediction_propagates_blank_image_error(self):
        """Verify that attempting prediction on blank image raises BlankImageError."""
        img = Image.new("L", (100, 100), color=255)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        with pytest.raises(BlankImageError):
            predict_digit(buf.getvalue())
