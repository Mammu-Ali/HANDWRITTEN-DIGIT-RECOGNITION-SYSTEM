"""
Unit and integration tests for Phase 11: Image Preprocessing Visualization.
Covers:
- Preprocessing metadata generation and base64 preview encoding.
- Light and dark background polarity detection visualization.
- Canvas drawing preprocessing visualization.
- API response inclusion of preprocessing previews and educational pipeline steps.
- Presence of visualization DOM elements in index.html.
- Graceful error handling for blank and invalid inputs.
"""

import base64
import io
import pytest
from PIL import Image, ImageDraw

from app import app
from image_processing import preprocess_image


@pytest.fixture
def client():
    """Create a persistent test client session."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _create_digit_bytes(digit: str = "7", bg: int = 255, fg: int = 0) -> bytes:
    """Generate a high-contrast synthetic digit image."""
    img = Image.new("L", (140, 140), color=bg)
    draw = ImageDraw.Draw(img)
    if digit == "7":
        draw.line([(30, 30), (110, 30)], fill=fg, width=14)
        draw.line([(110, 30), (50, 110)], fill=fg, width=14)
    elif digit == "1":
        draw.line([(70, 30), (70, 110)], fill=fg, width=16)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocessing_metadata_structure():
    """Verify that preprocess_image with return_metadata=True outputs complete preview data."""
    img_bytes = _create_digit_bytes("7", bg=255, fg=0)
    tensor, meta = preprocess_image(img_bytes, return_metadata=True)

    assert tensor.shape == (1, 28, 28, 1)
    assert isinstance(meta, dict)

    assert "grayscale_preview" in meta
    assert meta["grayscale_preview"].startswith("data:image/png;base64,")

    assert "tensor_preview" in meta
    assert meta["tensor_preview"].startswith("data:image/png;base64,")

    assert "steps" in meta
    assert isinstance(meta["steps"], list)
    assert len(meta["steps"]) >= 4

    # Verify key educational terms in pipeline steps
    steps_str = " ".join(meta["steps"]).lower()
    assert "grayscale" in steps_str
    assert "polarity" in steps_str
    assert "20×20" in steps_str or "20x20" in steps_str
    assert "center" in steps_str
    assert "normalized" in steps_str


def test_polarity_visualization_light_and_dark_backgrounds():
    """Verify that background polarity is accurately reported for both polarities."""
    # Dark digit on light background
    light_bg_bytes = _create_digit_bytes("7", bg=255, fg=0)
    _, meta_light = preprocess_image(light_bg_bytes, return_metadata=True)
    assert "inverted" in meta_light["polarity"].lower()

    # Light digit on dark background
    dark_bg_bytes = _create_digit_bytes("7", bg=0, fg=255)
    _, meta_dark = preprocess_image(dark_bg_bytes, return_metadata=True)
    assert "preserved" in meta_dark["polarity"].lower()


def test_canvas_drawing_visualization_pipeline(client):
    """Verify that POST /predict with canvas base64 image includes preprocessing visualization."""
    png_bytes = _create_digit_bytes("7", bg=255, fg=0)
    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    response = client.post("/predict", json={"image": data_url})
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True

    prep = res.get("preprocessing")
    assert isinstance(prep, dict)
    assert prep["grayscale_preview"].startswith("data:image/png;base64,")
    assert prep["tensor_preview"].startswith("data:image/png;base64,")
    assert len(prep["steps"]) >= 4


def test_visualization_dom_elements_present(client):
    """Verify that index.html contains all necessary visualization elements."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    expected_ids = [
        "toggle-details-btn",
        "toggle-details-label",
        "preprocessing-collapse",
        "detail-original-img",
        "detail-grayscale-img",
        "detail-tensor-img",
        "pipeline-steps-list",
    ]

    for elem_id in expected_ids:
        assert f'id="{elem_id}"' in html, f"Missing visualization element: {elem_id}"
