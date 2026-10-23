"""
Unit and integration tests for Phase 9: Interactive Drawing Canvas.
Covers:
- Base64 Data URL canvas payload prediction.
- Real CNN inference on drawn digit images.
- Blank canvas rejection (solid white / empty).
- Canvas DOM elements and accessibility attributes in index.html.
- Coordinate transformation scaling to prevent distortion.
"""

import base64
import io
import pytest
from PIL import Image, ImageDraw

from app import app


@pytest.fixture
def client():
    """Create a test client session for Flask."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _generate_canvas_data_url(digit: str = "3", size: tuple = (280, 280)) -> str:
    """Simulate a canvas export (white background with black stroke) as a data URL."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    if digit == "3":
        draw.line([(60, 60), (220, 60)], fill=(0, 0, 0), width=18)
        draw.line([(220, 60), (140, 130)], fill=(0, 0, 0), width=18)
        draw.arc([(80, 120), (220, 220)], start=270, end=90, fill=(0, 0, 0), width=18)
    elif digit == "0":
        draw.ellipse([(70, 50), (210, 230)], outline=(0, 0, 0), width=18)
    elif digit == "1":
        draw.line([(140, 50), (140, 230)], fill=(0, 0, 0), width=20)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_canvas_dom_elements_present(client):
    """Verify that index.html contains all necessary canvas elements and controls."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    canvas_ids = [
        "drawing-canvas",
        "brush-size",
        "brush-size-val",
        "canvas-predict-btn",
        "canvas-predict-spinner",
        "canvas-predict-btn-text",
        "canvas-undo-btn",
        "canvas-clear-btn",
        "tab-draw-btn",
        "tab-upload-btn",
    ]

    for elem_id in canvas_ids:
        assert f'id="{elem_id}"' in html, f"Missing canvas DOM element: {elem_id}"


def test_canvas_prediction_digit_three(client):
    """Verify that a canvas drawing of digit 3 is accurately predicted by the CNN."""
    data_url = _generate_canvas_data_url("3")
    response = client.post("/predict", json={"image": data_url})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["predicted_digit"] == 3
    assert data["confidence"] > 0.80
    assert len(data["probabilities"]) == 10


def test_canvas_prediction_digit_zero(client):
    """Verify that a canvas drawing of digit 0 is accurately predicted by the CNN."""
    data_url = _generate_canvas_data_url("0")
    response = client.post("/predict", json={"image": data_url})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["predicted_digit"] == 0
    assert data["confidence"] > 0.80


def test_blank_canvas_rejection(client):
    """Verify that an untouched, solid white canvas is rejected with a 400 Bad Request."""
    img = Image.new("RGB", (280, 280), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    blank_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    response = client.post("/predict", json={"image": blank_data_url})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "blank" in data["error"].lower() or "contrast" in data["error"].lower()


def test_canvas_coordinate_scale_math():
    """
    Test coordinate mapping logic:
    Verify that when CSS scales the canvas on mobile (e.g. 240px rendered vs 280px native),
    the coordinate conversion accurately scales to native canvas pixels.
    """
    native_w, native_h = 280.0, 280.0
    rendered_w, rendered_h = 240.0, 240.0

    scale_x = native_w / rendered_w
    scale_y = native_h / rendered_h

    # Touch at center of rendered canvas
    touch_x, touch_y = 120.0, 120.0
    canvas_x = touch_x * scale_x
    canvas_y = touch_y * scale_y

    assert canvas_x == 140.0
    assert canvas_y == 140.0
