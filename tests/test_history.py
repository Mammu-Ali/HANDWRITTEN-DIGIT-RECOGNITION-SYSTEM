"""
Unit and integration tests for Phase 12: Simple Prediction History.
Covers:
- Presence and accessibility of prediction history elements in index.html.
- CSS style definitions for history cards, active states, and thumbnails.
- Frontend script functions for history storage, validation, limit, and clearing.
- Backend API response structure completeness for constructing history items.
- Simulated multi-prediction session history serialization and FIFO trimming.
"""

import json
import io
import re
import pytest
from PIL import Image, ImageDraw

from app import app


@pytest.fixture
def client():
    """Create a persistent test client session."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _create_digit_bytes(digit: str = "3") -> bytes:
    """Generate high-contrast digit image bytes for testing."""
    img = Image.new("L", (140, 140), color=255)
    draw = ImageDraw.Draw(img)
    if digit == "3":
        draw.line([(30, 30), (110, 30)], fill=0, width=14)
        draw.line([(70, 70), (110, 70)], fill=0, width=14)
        draw.line([(30, 110), (110, 110)], fill=0, width=14)
        draw.line([(110, 30), (110, 110)], fill=0, width=14)
    elif digit == "1":
        draw.line([(70, 20), (70, 120)], fill=0, width=16)
    else:
        draw.rectangle([(40, 40), (100, 100)], outline=0, width=14)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_history_html_structure(client):
    """Verify that index.html contains all necessary history elements and state containers."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # Header and badges
    assert 'id="history-count-badge"' in html
    assert 'id="clear-history-btn"' in html
    assert "Recent Predictions History" in html

    # Empty state container
    assert 'id="history-empty"' in html
    assert "No predictions recorded in this browser session yet" in html

    # History items container
    assert 'id="history-items-container"' in html


def test_history_css_styling():
    """Verify that style.css contains styling for history cards, active state, and thumbnails."""
    with open("static/css/style.css", "r", encoding="utf-8") as f:
        css = f.read()

    assert ".history-item-card" in css
    assert ".active-history-card" in css
    assert ".history-thumb" in css
    assert ".history-digit-hero" in css
    assert "cursor: pointer" in css


def test_history_script_logic():
    """Verify that script.js defines history storage key, validation, saving, and rendering."""
    with open("static/js/script.js", "r", encoding="utf-8") as f:
        js = f.read()

    # Storage key & limits
    assert 'mnist_prediction_history' in js
    assert 'MAX_HISTORY_ITEMS' in js

    # Core history methods
    assert 'isValidHistoryItem' in js
    assert 'getStoredHistory' in js
    assert 'setStoredHistory' in js
    assert 'savePredictionToHistory' in js
    assert 'inspectHistoryItem' in js
    assert 'renderHistory' in js
    assert 'clearHistory' in js

    # Confirmation prompt
    assert 'confirm(' in js


def test_prediction_api_supplies_history_payload(client):
    """Verify that /predict returns all necessary fields required by the history schema."""
    img_bytes = _create_digit_bytes("3")
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(img_bytes), "digit3.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()

    assert data["success"] is True
    # Required for history item
    assert isinstance(data["predicted_digit"], int)
    assert 0 <= data["predicted_digit"] <= 9
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 10
    assert "top_three" in data
    assert len(data["top_three"]) == 3
    assert "preprocessing" in data
    assert "tensor_preview" in data["preprocessing"]


def test_simulated_history_fifo_trimming(client):
    """Simulate recording predictions and verify JSON serializability and FIFO maximum length."""
    history = []
    MAX_ITEMS = 10

    for i in range(15):
        img_bytes = _create_digit_bytes("1" if i % 2 == 0 else "3")
        response = client.post(
            "/predict",
            data={"file": (io.BytesIO(img_bytes), f"test_{i}.png", "image/png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        data = response.get_json()

        item = {
            "id": f"hist_{i}",
            "digit": data["predicted_digit"],
            "confidence": data["confidence"],
            "inputType": "Upload" if i % 2 == 0 else "Canvas",
            "timestamp": f"12:{i:02d}:00 PM",
            "thumbnail": data["preprocessing"]["tensor_preview"],
            "originalPreview": "data:image/png;base64,fake",
            "data": data,
        }

        # Prepend and trim to MAX_ITEMS
        history.insert(0, item)
        if len(history) > MAX_ITEMS:
            history = history[:MAX_ITEMS]

    assert len(history) == MAX_ITEMS
    # The most recent item should be at index 0
    assert history[0]["id"] == "hist_14"
    # Oldest retained item
    assert history[-1]["id"] == "hist_5"

    # Verify complete JSON roundtrip serialization
    serialized = json.dumps(history)
    deserialized = json.loads(serialized)
    assert len(deserialized) == MAX_ITEMS
    assert deserialized[0]["digit"] == history[0]["digit"]
    assert deserialized[0]["inputType"] == history[0]["inputType"]
