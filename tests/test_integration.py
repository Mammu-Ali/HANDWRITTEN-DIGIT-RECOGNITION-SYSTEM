"""
End-to-end integration tests for Phase 6.
Verifies the complete workflow:
- Multiple dynamic predictions with real model inferences (ensuring responses are not hardcoded).
- Repetitive predictions on a single persistent client session.
- Response schema compliance with frontend validation rules.
- Proper handling of edge cases and error messages.
"""

import io
import json
import pytest
from PIL import Image, ImageDraw

from app import app


@pytest.fixture
def client():
    """Create a persistent test client session."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _draw_digit_image(digit_char: str) -> bytes:
    """Generate high-contrast digit images for end-to-end tests."""
    img = Image.new("L", (140, 140), color=255)
    draw = ImageDraw.Draw(img)

    if digit_char == "7":
        draw.line([(30, 30), (110, 30)], fill=0, width=14)
        draw.line([(110, 30), (50, 110)], fill=0, width=14)
    elif digit_char == "1":
        draw.line([(70, 30), (70, 110)], fill=0, width=16)
    elif digit_char == "0":
        draw.ellipse([(40, 30), (100, 110)], outline=0, width=14)
    else:
        draw.line([(30, 70), (110, 70)], fill=0, width=14)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_repeated_predictions_dynamic_inference(client):
    """
    Test repeated predictions on a single session.
    Verifies that different input images return different predicted digits,
    confirming predictions are dynamically computed by the CNN rather than hardcoded.
    """
    # 1. Predict Digit 7
    png_seven = _draw_digit_image("7")
    res1 = client.post(
        "/predict",
        data={"file": (io.BytesIO(png_seven), "digit7.png")},
        content_type="multipart/form-data",
    )
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert data1["success"] is True
    assert data1["predicted_digit"] == 7
    assert data1["confidence"] > 0.80

    # 2. Predict Digit 1 (repeated prediction without reload)
    png_one = _draw_digit_image("1")
    res2 = client.post(
        "/predict",
        data={"file": (io.BytesIO(png_one), "digit1.png")},
        content_type="multipart/form-data",
    )
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["success"] is True
    assert data2["predicted_digit"] == 1
    assert data2["confidence"] > 0.80

    # Confirm that outputs differ between distinct inputs
    assert data1["predicted_digit"] != data2["predicted_digit"]
    assert data1["probabilities"]["7"] > data1["probabilities"]["1"]
    assert data2["probabilities"]["1"] > data2["probabilities"]["7"]


def test_frontend_schema_compliance(client):
    """
    Verify that the payload structure satisfies every check performed
    by the frontend controller's isValidPredictionResponse() function.
    """
    png_seven = _draw_digit_image("7")
    res = client.post(
        "/predict",
        data={"file": (io.BytesIO(png_seven), "test.png")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    data = res.get_json()

    # Front-end validation rules:
    assert isinstance(data, dict)
    assert data.get("success") is True
    assert isinstance(data.get("predicted_digit"), int)
    assert 0 <= data["predicted_digit"] <= 9
    assert isinstance(data.get("confidence"), float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data.get("probabilities"), dict)
    assert len(data["probabilities"]) == 10

    for digit in range(10):
        digit_key = str(digit)
        assert digit_key in data["probabilities"]
        val = data["probabilities"][digit_key]
        assert isinstance(val, (int, float))
        assert 0.0 <= val <= 1.0


def test_error_flow_integration(client):
    """Verify that errors return clean JSON with user-readable error strings."""
    # Blank image
    blank = Image.new("L", (100, 100), color=255)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")

    res = client.post(
        "/predict",
        data={"file": (io.BytesIO(buf.getvalue()), "blank.png")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert len(data["error"]) > 0
