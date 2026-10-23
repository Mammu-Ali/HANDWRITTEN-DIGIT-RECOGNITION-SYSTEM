"""
Unit and integration tests for Phase 10: Enhanced Prediction Results.
Covers:
- Prominent predicted digit and top-class confidence.
- Correctly sorted Top-3 classes by descending probability.
- Probability distribution for digits 0 to 9.
- Both canvas and file upload prediction pathways.
- Presence of Top-3 and Predict Again DOM elements.
"""

import base64
import io
import pytest
from PIL import Image, ImageDraw

from app import app


@pytest.fixture
def client():
    """Create a persistent test client session."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _generate_synthetic_digit_png(digit: str = "7") -> bytes:
    """Generate a clean synthetic digit PNG image in memory."""
    img = Image.new("L", (140, 140), color=255)
    draw = ImageDraw.Draw(img)
    if digit == "7":
        draw.line([(30, 30), (110, 30)], fill=0, width=14)
        draw.line([(110, 30), (50, 110)], fill=0, width=14)
    elif digit == "3":
        draw.line([(30, 30), (110, 30)], fill=0, width=14)
        draw.line([(110, 30), (70, 65)], fill=0, width=14)
        draw.arc([(40, 60), (110, 110)], start=270, end=90, fill=0, width=14)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_enhanced_results_dom_elements(client):
    """Verify that index.html contains elements for Top-3 results and Predict Again action."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    assert 'id="top-three-list"' in html
    assert 'id="predict-again-btn"' in html
    assert 'id="result-digit"' in html
    assert 'id="result-confidence"' in html
    assert 'id="probabilities-list"' in html


def test_top_three_sorted_correctly_multipart(client):
    """
    Verify that /predict returns top_three sorted in strict descending order,
    and that 1st place corresponds to the predicted digit.
    """
    png_bytes = _generate_synthetic_digit_png("7")
    data = {"file": (io.BytesIO(png_bytes), "digit7.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True

    top_three = res.get("top_three")
    assert isinstance(top_three, list)
    assert len(top_three) == 3

    # Check descending order: p[0] >= p[1] >= p[2]
    assert top_three[0]["probability"] >= top_three[1]["probability"]
    assert top_three[1]["probability"] >= top_three[2]["probability"]

    # Check 1st candidate matches the primary predicted digit
    assert top_three[0]["digit"] == res["predicted_digit"]
    assert top_three[0]["digit"] == 7

    # Check probabilities dict consistency
    for item in top_three:
        digit_key = str(item["digit"])
        assert digit_key in res["probabilities"]
        assert item["probability"] == res["probabilities"][digit_key]


def test_top_three_sorted_correctly_canvas_json(client):
    """
    Verify top-3 sorting and probability distribution for a canvas base64 drawing request.
    """
    png_bytes = _generate_synthetic_digit_png("3")
    base64_str = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    response = client.post("/predict", json={"image": base64_str})
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True

    top_three = res.get("top_three")
    assert isinstance(top_three, list)
    assert len(top_three) == 3

    # Strict descending sort
    assert top_three[0]["probability"] >= top_three[1]["probability"]
    assert top_three[1]["probability"] >= top_three[2]["probability"]

    # 1st rank matches primary predicted digit
    assert top_three[0]["digit"] == res["predicted_digit"]
    assert top_three[0]["digit"] == 3


def test_probability_values_consistency(client):
    """Verify that all 10 probabilities are properly bounded and sum to ~1.0."""
    png_bytes = _generate_synthetic_digit_png("7")
    data = {"file": (io.BytesIO(png_bytes), "digit7.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    probs = res["probabilities"]

    assert len(probs) == 10
    total = sum(probs.values())
    assert pytest.approx(total, rel=1e-2) == 1.0

    for digit in range(10):
        val = probs[str(digit)]
        assert 0.0 <= val <= 1.0
