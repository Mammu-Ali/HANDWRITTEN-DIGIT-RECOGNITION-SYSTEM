"""
Unit and integration tests for Phase 14 / 16: Model Information Page.
Covers:
- /model-info endpoint status and content.
- Accuracy, loss, parameter count in evaluation artifact.
- Confusion matrix dimensions and non-fabrication checks.
- Per-class metrics (precision, recall, f1-score).
- Graceful error notification when metrics artifact is missing.
"""

import json
import os
import pytest

from app import app


@pytest.fixture
def client():
    """Create a test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_model_info_page_renders_successfully(client):
    """Verify that /model-info returns 200 OK with expected headings."""
    response = client.get("/model-info")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "CNN Model Architecture &amp; Evaluation" in html or "CNN Model Architecture & Evaluation" in html
    assert "Test Accuracy" in html
    assert "Test Loss" in html
    assert "Layer Architecture Summary" in html
    assert "Confusion Matrix" in html
    assert "Per-Class Classification Metrics" in html


def test_evaluation_metrics_artifact_validity():
    """Verify the genuine evaluation metrics in evaluation_metrics.json."""
    assert os.path.exists("evaluation_metrics.json"), "evaluation_metrics.json must be present."

    with open("evaluation_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Core checks
    assert metrics["test_loss"] == 0.0262
    assert metrics["test_accuracy"] == 0.991
    assert metrics["test_samples"] == 10000
    assert metrics["total_parameters"] > 100000

    # 10x10 Confusion matrix
    cm = metrics["confusion_matrix"]
    assert len(cm) == 10
    assert all(len(row) == 10 for row in cm)
    total_evaluated = sum(sum(row) for row in cm)
    assert total_evaluated == 10000

    # Class metrics
    assert len(metrics["class_metrics"]) == 10
    for digit_str in map(str, range(10)):
        cm_digit = metrics["class_metrics"][digit_str]
        assert 0.95 <= cm_digit["precision"] <= 1.0
        assert 0.95 <= cm_digit["recall"] <= 1.0
        assert 0.95 <= cm_digit["f1_score"] <= 1.0
        assert cm_digit["support"] > 800


def test_model_info_missing_artifact_graceful_handling(client, monkeypatch):
    """Verify that /model-info gracefully shows a warning when evaluation artifact is missing."""
    monkeypatch.setattr(os.path, "exists", lambda path: False if "evaluation_metrics.json" in path else os.path.exists(path))

    response = client.get("/model-info")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Evaluation Artifact Notice" in html or "not found" in html
