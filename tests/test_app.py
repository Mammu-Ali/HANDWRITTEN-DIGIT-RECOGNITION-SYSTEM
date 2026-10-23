"""
Unit, integration, and regression tests for the Handwritten Digit Recognition Flask application.
Phase 7 Quality Assurance suite covering:
1. Home page and health endpoint.
2. Valid PNG and JPEG predictions.
3. Unsupported, corrupted, and empty images.
4. Missing files and empty JSON requests.
5. Oversized upload rejection (HTTP 413).
6. Blank-image handling (white and black blank images).
7. Valid predicted digit from 0–9.
8. Ten probability values within valid ranges summing to 1.0.
9. Correct image preprocessing dimensions and normalization.
10. Missing or unreadable model file handling (HTTP 503).
11. Frontend template structure, required DOM elements, and static assets.
12. Regression tests: transparency handling, reverse polarity, and data URL parsing.
"""

import base64
import io
import pytest
from PIL import Image, ImageDraw

from app import app
from predictor import DigitPredictor


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _create_synthetic_digit(
    digit: str = "7",
    img_format: str = "PNG",
    bg_color: int = 255,
    fg_color: int = 0,
    mode: str = "L",
) -> bytes:
    """Generate a clean synthetic digit image in memory."""
    img = Image.new(mode, (140, 140), color=bg_color)
    draw = ImageDraw.Draw(img)

    if digit == "7":
        draw.line([(30, 30), (110, 30)], fill=fg_color, width=14)
        draw.line([(110, 30), (50, 110)], fill=fg_color, width=14)
    elif digit == "1":
        draw.line([(70, 30), (70, 110)], fill=fg_color, width=16)
    elif digit == "0":
        draw.ellipse([(40, 30), (100, 110)], outline=fg_color, width=14)
    else:
        draw.line([(30, 70), (110, 70)], fill=fg_color, width=14)

    buf = io.BytesIO()
    img.save(buf, format=img_format)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Home Page and Health Endpoint Tests
# ---------------------------------------------------------------------------

def test_home_page(client):
    """Verify that the home page loads successfully with status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Handwritten Digit Recognition" in response.data
    assert b"MNIST Deep Learning Classification System" in response.data


def test_health_check(client):
    """Verify that the /health endpoint returns HTTP 200 and reports model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data is not None
    assert json_data.get("status") == "ok"
    assert json_data.get("app") == "Handwritten Digit Recognition"
    assert json_data.get("model_loaded") is True


# ---------------------------------------------------------------------------
# 2. Valid PNG and JPEG Predictions
# ---------------------------------------------------------------------------

def test_predict_valid_png_multipart(client):
    """Verify POST /predict with multipart/form-data PNG image."""
    png_bytes = _create_synthetic_digit("7", img_format="PNG")
    data = {"file": (io.BytesIO(png_bytes), "digit7.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["predicted_digit"] == 7
    assert json_data["confidence"] > 0.80


def test_predict_valid_jpeg_multipart(client):
    """Verify POST /predict with multipart/form-data JPEG image."""
    jpeg_bytes = _create_synthetic_digit("1", img_format="JPEG", mode="RGB", bg_color=(255, 255, 255), fg_color=(0, 0, 0))
    data = {"file": (io.BytesIO(jpeg_bytes), "digit1.jpg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["predicted_digit"] == 1
    assert json_data["confidence"] > 0.80


def test_predict_valid_image_json_base64(client):
    """Verify POST /predict with base64 Data URL in JSON payload."""
    png_bytes = _create_synthetic_digit("7", img_format="PNG")
    base64_str = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    response = client.post("/predict", json={"image": base64_str})
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["predicted_digit"] == 7
    assert json_data["confidence"] > 0.80


# ---------------------------------------------------------------------------
# 3. Unsupported and Corrupted Images
# ---------------------------------------------------------------------------

def test_predict_unsupported_format(client):
    """Verify POST /predict with unsupported image format (BMP) returns 400."""
    bmp_img = Image.new("RGB", (60, 60), color=(255, 255, 255))
    buf = io.BytesIO()
    bmp_img.save(buf, format="BMP")

    data = {"file": (io.BytesIO(buf.getvalue()), "digit.bmp")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "unsupported" in json_data["error"].lower()


def test_predict_corrupted_file(client):
    """Verify POST /predict with corrupt byte stream returns 400 Bad Request."""
    data = {"file": (io.BytesIO(b"not an image byte stream!"), "corrupt.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "decode" in json_data["error"].lower() or "failed" in json_data["error"].lower()


# ---------------------------------------------------------------------------
# 4. Missing Files and Empty Requests
# ---------------------------------------------------------------------------

def test_predict_missing_file_multipart(client):
    """Verify POST /predict without file or JSON body returns 400 Bad Request."""
    response = client.post("/predict")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "no image provided" in json_data["error"].lower()


def test_predict_empty_file(client):
    """Verify POST /predict with 0-byte file upload returns 400 Bad Request."""
    data = {"file": (io.BytesIO(b""), "empty.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "empty" in json_data["error"].lower()


def test_predict_empty_json_body(client):
    """Verify POST /predict with empty JSON body {} returns 400 Bad Request."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "no valid 'image' field" in json_data["error"].lower()


# ---------------------------------------------------------------------------
# 5. Oversized Upload Rejection
# ---------------------------------------------------------------------------

def test_predict_oversized_request(client):
    """Verify POST /predict with payload exceeding 2MB returns 413 Payload Too Large."""
    oversized_data = b"X" * (3 * 1024 * 1024)  # 3 MB
    data = {"file": (io.BytesIO(oversized_data), "huge.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 413
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "limit" in json_data["error"].lower() or "exceeds" in json_data["error"].lower()


# ---------------------------------------------------------------------------
# 6. Blank Image Handling
# ---------------------------------------------------------------------------

def test_predict_blank_white_image(client):
    """Verify POST /predict with solid white image returns 400 with helpful error."""
    blank = Image.new("L", (100, 100), color=255)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")

    data = {"file": (io.BytesIO(buf.getvalue()), "blank_white.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "blank" in json_data["error"].lower() or "contrast" in json_data["error"].lower()


def test_predict_blank_black_image(client):
    """Verify POST /predict with solid black image returns 400 with helpful error."""
    blank = Image.new("L", (100, 100), color=0)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")

    data = {"file": (io.BytesIO(buf.getvalue()), "blank_black.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "blank" in json_data["error"].lower() or "contrast" in json_data["error"].lower()


# ---------------------------------------------------------------------------
# 7 & 8. Valid Predicted Digit (0-9) & Probability Ranges
# ---------------------------------------------------------------------------

def test_prediction_digit_and_probabilities_schema(client):
    """
    Verify predicted_digit is in range [0, 9] and that all 10 probabilities
    are valid floats in [0.0, 1.0] whose sum is approximately 1.0.
    """
    png_bytes = _create_synthetic_digit("7")
    data = {"file": (io.BytesIO(png_bytes), "digit7.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_data = response.get_json()

    predicted_digit = json_data["predicted_digit"]
    assert isinstance(predicted_digit, int)
    assert 0 <= predicted_digit <= 9

    probabilities = json_data["probabilities"]
    assert isinstance(probabilities, dict)
    assert len(probabilities) == 10
    assert set(probabilities.keys()) == {str(i) for i in range(10)}

    prob_sum = 0.0
    for digit_str, prob_val in probabilities.items():
        assert isinstance(prob_val, float)
        assert 0.0 <= prob_val <= 1.0
        prob_sum += prob_val

    assert pytest.approx(prob_sum, rel=1e-2) == 1.0

    # Ensure predicted digit corresponds to highest probability
    highest_prob_digit = int(max(probabilities, key=probabilities.get))
    assert predicted_digit == highest_prob_digit


# ---------------------------------------------------------------------------
# 9. Correct Image Preprocessing Dimensions
# ---------------------------------------------------------------------------

def test_preprocessing_dimensions_via_module():
    """Verify preprocess_image output tensor shape is strictly (1, 28, 28, 1)."""
    from image_processing import preprocess_image
    png_bytes = _create_synthetic_digit("7")
    tensor = preprocess_image(png_bytes)
    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == "float32"


# ---------------------------------------------------------------------------
# 10. Missing or Unreadable Model File Handling
# ---------------------------------------------------------------------------

def test_missing_model_file_handling():
    """Verify that initializing DigitPredictor with missing model raises FileNotFoundError."""
    predictor = DigitPredictor(model_path="nonexistent_model_weights.keras")
    with pytest.raises(FileNotFoundError) as exc_info:
        _ = predictor.model
    assert "not found" in str(exc_info.value).lower()


def test_service_unavailable_when_model_is_none(client, monkeypatch):
    """Verify that /predict returns HTTP 503 if the model service is unavailable."""
    import app as app_module
    monkeypatch.setattr(app_module, "predictor", None)

    png_bytes = _create_synthetic_digit("7")
    data = {"file": (io.BytesIO(png_bytes), "digit7.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 503
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "unavailable" in json_data["error"].lower()


# ---------------------------------------------------------------------------
# 11. Frontend Template Structure and Static Assets
# ---------------------------------------------------------------------------

def test_frontend_dom_elements_present(client):
    """
    Verify that the rendered index.html contains all necessary element IDs
    and accessible markup required by script.js.
    """
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    required_ids = [
        "drop-zone",
        "file-input",
        "browse-btn",
        "drop-prompt",
        "preview-wrapper",
        "image-preview",
        "file-details",
        "predict-btn",
        "predict-btn-text",
        "predict-spinner",
        "clear-btn",
        "api-status-badge",
        "state-empty",
        "state-loading",
        "state-error",
        "state-success",
        "error-message",
        "result-digit",
        "result-confidence",
        "probabilities-list",
    ]

    for element_id in required_ids:
        assert f'id="{element_id}"' in html, f"Missing required DOM element id: {element_id}"


def test_static_assets_delivered(client):
    """Verify that static CSS and JavaScript files return HTTP 200."""
    css_res = client.get("/static/css/style.css")
    assert css_res.status_code == 200
    assert b"--accent-blue" in css_res.data

    js_res = client.get("/static/js/script.js")
    assert js_res.status_code == 200
    assert b"submitPrediction" in js_res.data


# ---------------------------------------------------------------------------
# 12. Regression Tests
# ---------------------------------------------------------------------------

def test_regression_transparent_rgba_image(client):
    """
    Regression test: RGBA images with transparency (such as HTML5 canvas exports)
    must composite cleanly onto a white background rather than causing errors or noise.
    """
    img = Image.new("RGBA", (140, 140), color=(255, 255, 255, 0))  # Transparent
    draw = ImageDraw.Draw(img)
    # Draw opaque black digit 1
    draw.line([(70, 30), (70, 110)], fill=(0, 0, 0, 255), width=16)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    data = {"file": (io.BytesIO(buf.getvalue()), "transparent_digit.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["predicted_digit"] == 1


def test_regression_reverse_polarity_dark_canvas(client):
    """
    Regression test: A white digit drawn on a dark background must not be
    inverted erroneously, and should be correctly recognized.
    """
    png_bytes = _create_synthetic_digit("7", bg_color=0, fg_color=255)
    data = {"file": (io.BytesIO(png_bytes), "white_on_black.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["predicted_digit"] == 7


def test_regression_method_not_allowed(client):
    """Verify GET /predict returns 405 Method Not Allowed."""
    response = client.get("/predict")
    assert response.status_code == 405
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "Method not allowed" in json_data["error"]


def test_regression_no_stack_trace_leakage(client):
    """Verify error responses never expose local paths or tracebacks."""
    response = client.post("/predict", json={"image": "invalid_base64!!!"})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    error_text = json_data["error"]
    assert "Traceback" not in error_text
    assert "d:\\" not in error_text.lower()
    assert "c:\\" not in error_text.lower()
