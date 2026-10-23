"""
Unit and integration tests for Phase 13 / 16: Sample Digit Gallery.
Covers:
- Presence and validity of all ten sample digit images (0-9).
- Image dimensions and PNG formatting.
- Image serving endpoint /samples/<filename>.
- Metadata API endpoint /api/samples.
- Real model prediction accuracy across all 10 sample digits.
- Presence of gallery DOM elements in index.html.
"""

import io
import os
import pytest
from PIL import Image

from app import app
from predictor import get_default_predictor


@pytest.fixture
def client():
    """Create a test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_sample_files_exist_and_are_valid():
    """Verify that all ten sample digit images exist and are readable PNGs."""
    samples_dir = "samples"
    assert os.path.exists(samples_dir)

    for digit in range(10):
        filename = f"sample_digit_{digit}.png"
        filepath = os.path.join(samples_dir, filename)
        assert os.path.exists(filepath), f"Sample file '{filename}' missing."

        with Image.open(filepath) as img:
            assert img.format == "PNG"
            assert img.size[0] > 0 and img.size[1] > 0


def test_serve_sample_endpoint(client):
    """Verify that /samples/<filename> serves each image correctly."""
    for digit in range(10):
        filename = f"sample_digit_{digit}.png"
        response = client.get(f"/samples/{filename}")
        assert response.status_code == 200
        assert "image/png" in response.content_type
        assert len(response.data) > 0


def test_api_samples_endpoint(client):
    """Verify that /api/samples returns full metadata for digits 0-9."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.get_json()

    assert data["success"] is True
    assert "samples" in data
    assert len(data["samples"]) == 10

    for i, sample in enumerate(data["samples"]):
        assert sample["digit"] == i
        assert sample["filename"] == f"sample_digit_{i}.png"
        assert sample["url"] == f"/samples/sample_digit_{i}.png"
        assert sample["label"] == f"Digit {i}"


def test_sample_predictions_with_trained_model(client):
    """Verify that submitting sample images to /predict yields expected digits using the real model."""
    for digit in range(10):
        with open(f"samples/sample_digit_{digit}.png", "rb") as f:
            img_bytes = f.read()

        response = client.post(
            "/predict",
            data={"file": (io.BytesIO(img_bytes), f"sample_digit_{digit}.png", "image/png")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 200
        result = response.get_json()
        assert result["success"] is True
        assert result["predicted_digit"] == digit
        assert result["confidence"] > 0.5


def test_gallery_dom_elements_present(client):
    """Verify that index.html contains tab controls and gallery cards."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert 'id="tab-gallery-btn"' in html
    assert 'id="tab-gallery-pane"' in html
    assert 'id="gallery-selected-badge"' in html
    assert 'id="gallery-predict-btn"' in html
    assert 'id="gallery-clear-btn"' in html
    assert "sample-digit-card" in html

    for digit in range(10):
        assert f'data-digit="{digit}"' in html
        assert f"Ground Truth: {digit}" in html
