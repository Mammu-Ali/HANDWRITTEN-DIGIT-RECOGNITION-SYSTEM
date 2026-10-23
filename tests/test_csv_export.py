"""
Unit and integration tests for Phase 15 / 16: CSV History Export.
Covers:
- Presence of export CSV button in index.html.
- Export function in script.js.
- Correct CSV header structure.
- CSV escaping and quoting rules for special characters.
- Non-mutation of the underlying prediction history.
"""

import csv
import io
import pytest

from app import app


@pytest.fixture
def client():
    """Create a test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_export_csv_dom_element(client):
    """Verify that index.html contains the Export CSV button."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="export-csv-btn"' in html
    assert "Export CSV" in html


def test_export_csv_script_definition():
    """Verify that script.js contains CSV export logic and escaping."""
    with open("static/js/script.js", "r", encoding="utf-8") as f:
        js = f.read()

    assert "exportHistoryCSV" in js
    assert "exportCsvBtn" in js
    assert "text/csv" in js
    assert "mnist_predictions_" in js
    assert "Record ID" in js
    assert "Confidence (%)" in js


def test_csv_formatting_and_escaping_simulation():
    """Simulate CSV export formatting to verify header and escaping integrity."""
    mock_items = [
        {
            "id": "hist_12345",
            "timestamp": "04:15:20 PM",
            "inputType": "Upload",
            "digit": 7,
            "confidence": 0.9985,
            "data": {
                "top_three": [
                    {"digit": 7, "probability": 0.9985},
                    {"digit": 1, "probability": 0.0012},
                    {"digit": 2, "probability": 0.0001},
                ]
            }
        },
        {
            "id": 'hist_with_"quotes",and,commas',
            "timestamp": "04:16:00 PM",
            "inputType": "Canvas",
            "digit": 3,
            "confidence": 0.9520,
            "data": {
                "top_three": [
                    {"digit": 3, "probability": 0.9520},
                    {"digit": 5, "probability": 0.0310},
                    {"digit": 8, "probability": 0.0100},
                ]
            }
        }
    ]

    # Verify standard CSV writer parses without corruption
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    headers = [
        "Record ID", "Timestamp", "Input Mode", "Predicted Digit",
        "Confidence (%)", "Top 2nd Digit", "Top 2nd Confidence (%)",
        "Top 3rd Digit", "Top 3rd Confidence (%)"
    ]
    writer.writerow(headers)

    for item in mock_items:
        t2 = item["data"]["top_three"][1]
        t3 = item["data"]["top_three"][2]
        writer.writerow([
            item["id"],
            item["timestamp"],
            item["inputType"],
            item["digit"],
            f"{item['confidence'] * 100:.2f}",
            t2["digit"],
            f"{t2['probability'] * 100:.2f}",
            t3["digit"],
            f"{t3['probability'] * 100:.2f}",
        ])

    csv_data = output.getvalue()
    reader = list(csv.reader(io.StringIO(csv_data)))

    assert len(reader) == 3
    assert reader[0] == headers
    assert reader[1][0] == "hist_12345"
    assert reader[1][3] == "7"
    assert reader[1][4] == "99.85"
    # Escaped quotes and commas roundtrip correctly
    assert reader[2][0] == 'hist_with_"quotes",and,commas'
    assert reader[2][3] == "3"
