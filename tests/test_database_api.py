"""
Unit and integration tests for SQLite Database, History API, Feedback API, and CSV Export.
"""

import io
import json
import pytest
from PIL import Image, ImageDraw

from app import app
import database


@pytest.fixture
def client():
    """Create a persistent test client session."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _create_digit_image_bytes(digit: str = "7") -> bytes:
    """Generate high-contrast digit image for prediction."""
    img = Image.new("L", (140, 140), color=255)
    draw = ImageDraw.Draw(img)
    if digit == "7":
        draw.line([(30, 30), (110, 30)], fill=0, width=14)
        draw.line([(110, 30), (50, 110)], fill=0, width=14)
    elif digit == "1":
        draw.line([(70, 20), (70, 120)], fill=0, width=16)
    else:
        draw.ellipse([(30, 30), (110, 110)], outline=0, width=14)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_persists_to_sqlite(client):
    """Verify that calling /predict writes a record into SQLite database."""
    # First clear history
    client.delete("/api/history")

    png_bytes = _create_digit_image_bytes("7")
    data = {"file": (io.BytesIO(png_bytes), "digit7.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["success"] is True
    assert "prediction_id" in res_json
    pred_id = res_json["prediction_id"]

    # Verify query via /api/history
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    hist_json = hist_res.get_json()
    assert hist_json["success"] is True
    assert hist_json["count"] >= 1
    found = any(r["id"] == pred_id for r in hist_json["history"])
    assert found is True

    # Verify query via /api/history/<id>
    single_res = client.get(f"/api/history/{pred_id}")
    assert single_res.status_code == 200
    single_json = single_res.get_json()
    assert single_json["success"] is True
    assert single_json["prediction"]["id"] == pred_id
    assert single_json["prediction"]["predicted_digit"] == res_json["predicted_digit"]


def test_history_filtering_and_sorting(client):
    """Verify filtering by digit, input_type, and sorting."""
    # Insert known test records directly
    p1 = database.insert_prediction(
        predicted_digit=3,
        confidence=0.92,
        top_three=[{"digit": 3, "probability": 0.92}],
        probabilities={"3": 0.92},
        input_type="Canvas"
    )
    p2 = database.insert_prediction(
        predicted_digit=8,
        confidence=0.88,
        top_three=[{"digit": 8, "probability": 0.88}],
        probabilities={"8": 0.88},
        input_type="Upload"
    )

    # Filter by digit 3
    res_d3 = client.get("/api/history?digit=3")
    assert res_d3.status_code == 200
    items = res_d3.get_json()["history"]
    assert all(r["predicted_digit"] == 3 for r in items)

    # Filter by input_type Canvas
    res_canvas = client.get("/api/history?input_type=Canvas")
    assert res_canvas.status_code == 200
    items_c = res_canvas.get_json()["history"]
    assert all(r["input_type"] == "Canvas" for r in items_c)

    # Sort by confidence asc
    res_sort = client.get("/api/history?sort_by=confidence&sort_order=asc")
    assert res_sort.status_code == 200
    items_s = res_sort.get_json()["history"]
    if len(items_s) >= 2:
        assert items_s[0]["confidence"] <= items_s[1]["confidence"]


def test_feedback_api(client):
    """Verify submitting user verification feedback and retrieving stats."""
    p_id = database.insert_prediction(
        predicted_digit=5,
        confidence=0.95,
        top_three=[{"digit": 5, "probability": 0.95}],
        probabilities={"5": 0.95},
        input_type="Canvas"
    )

    # Submit feedback 'correct'
    fb_res = client.post("/api/feedback", json={
        "prediction_id": p_id,
        "feedback": "correct"
    })
    assert fb_res.status_code == 200
    fb_json = fb_res.get_json()
    assert fb_json["success"] is True

    # Check stats
    stats_res = client.get("/api/feedback/stats")
    assert stats_res.status_code == 200
    stats = stats_res.get_json()["stats"]
    assert stats["correct_count"] >= 1
    assert stats["observed_accuracy"] is not None

    # Submit feedback 'incorrect' with actual_digit
    fb_res2 = client.post("/api/feedback", json={
        "prediction_id": p_id,
        "feedback": "incorrect",
        "actual_digit": 6
    })
    assert fb_res2.status_code == 200

    record = database.get_prediction(p_id)
    assert record["feedback"] == "incorrect"
    assert record["actual_digit"] == 6


def test_history_csv_export_endpoint(client):
    """Verify GET /api/history/export returns downloadable CSV."""
    res = client.get("/api/history/export")
    assert res.status_code == 200
    assert "text/csv" in res.content_type
    assert "Record ID" in res.get_data(as_text=True)
    assert "attachment" in res.headers.get("Content-Disposition", "")


def test_delete_history_endpoints(client):
    """Verify deleting single record and clearing entire history."""
    p_id = database.insert_prediction(
        predicted_digit=2,
        confidence=0.85,
        top_three=[{"digit": 2, "probability": 0.85}],
        probabilities={"2": 0.85},
        input_type="Canvas"
    )

    del_res = client.delete(f"/api/history/{p_id}")
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True

    # Verify not found now
    check_res = client.get(f"/api/history/{p_id}")
    assert check_res.status_code == 404

    # Clear all
    clear_res = client.delete("/api/history")
    assert clear_res.status_code == 200
    assert clear_res.get_json()["success"] is True
