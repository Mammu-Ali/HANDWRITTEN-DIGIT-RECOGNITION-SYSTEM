"""
Flask Application for Handwritten Digit Recognition.
Phase 4: Robust REST API for digit prediction with image validation,
error handling, and model preloading.
"""

import base64
import binascii
import json
import logging
import os
import time
from typing import Tuple
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

import database
from image_processing import (
    BlankImageError,
    InvalidImageError,
    UnsupportedFormatError,
)
from predictor import get_default_predictor

# Maximum upload size: 2 Megabytes
MAX_CONTENT_LENGTH = 2 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Initialize SQLite database schema
database.init_db()

@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preload the predictor and model once during application initialization
try:
    predictor = get_default_predictor()
    _ = predictor.model  # Preload weights into memory
    logger.info("Digit model successfully loaded into memory.")
except Exception as init_err:
    logger.error("Failed to preload model at startup: %s", init_err)
    predictor = None


def extract_image_bytes(req) -> Tuple[bytes, str]:
    """
    Extract image bytes from either multipart/form-data file upload
    or JSON base64-encoded payload, along with detected input mode.
    """
    # 1. Check multipart/form-data upload
    if req.files:
        file_obj = req.files.get("file") or req.files.get("image")
        if not file_obj:
            # Fall back to first available uploaded file if key differs
            file_obj = next(iter(req.files.values()), None)

        if not file_obj or not file_obj.filename:
            raise InvalidImageError("No image file selected for upload.")

        file_bytes = file_obj.read()
        if len(file_bytes) == 0:
            raise InvalidImageError("Uploaded file is empty.")
        if len(file_bytes) > MAX_CONTENT_LENGTH:
            raise RequestEntityTooLarge("File size exceeds 2MB limit.")

        filename_lower = file_obj.filename.lower()
        if "sample_digit_" in filename_lower:
            input_mode = "Sample"
        else:
            input_mode = req.form.get("input_type", "Upload")
        return file_bytes, input_mode

    # 2. Check JSON payload (base64 data URL or raw base64)
    if req.is_json:
        data = req.get_json(silent=True) or {}
        image_data = data.get("image")
        if not image_data or not isinstance(image_data, str):
            raise InvalidImageError("No valid 'image' field found in JSON payload.")

        # Strip Data URL prefix if present (e.g. data:image/png;base64,...)
        if "," in image_data:
            _, image_data = image_data.split(",", 1)

        try:
            file_bytes = base64.b64decode(image_data)
        except (binascii.Error, ValueError) as err:
            raise InvalidImageError("Invalid base64 encoding.") from err

        if len(file_bytes) == 0:
            raise InvalidImageError("Decoded base64 image data is empty.")
        if len(file_bytes) > MAX_CONTENT_LENGTH:
            raise RequestEntityTooLarge("File size exceeds 2MB limit.")

        input_mode = data.get("input_type", "Canvas")
        return file_bytes, input_mode

    raise InvalidImageError(
        "No image provided. Please submit an image file via form-data or a base64 string via JSON."
    )


@app.route("/")
def index():
    """Render the main user interface."""
    return render_template("index.html")


@app.route("/model-info")
def model_info():
    """Render the Model Information and Evaluation Dashboard."""
    metrics_path = os.path.join(app.root_path, "evaluation_metrics.json")
    metrics_data = None
    metrics_error = None

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            metrics_error = f"Error reading evaluation metrics: {str(e)}"
    else:
        metrics_error = "Evaluation metrics artifact ('evaluation_metrics.json') not found. Please run evaluation script to generate it."

    is_model_ready = predictor is not None and predictor._model is not None
    return render_template(
        "model_info.html",
        metrics=metrics_data,
        metrics_error=metrics_error,
        model_ready=is_model_ready
    )


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    """Serve sample digit images for the Sample Gallery."""
    samples_dir = os.path.join(app.root_path, "samples")
    return send_from_directory(samples_dir, filename)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return list of available sample digit images."""
    samples = []
    for digit in range(10):
        filename = f"sample_digit_{digit}.png"
        samples.append({
            "digit": digit,
            "filename": filename,
            "url": f"/samples/{filename}",
            "label": f"Digit {digit}"
        })
    return jsonify({"success": True, "samples": samples}), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to monitor application and model status."""
    is_model_ready = predictor is not None and predictor._model is not None
    return jsonify({
        "status": "ok",
        "app": "Handwritten Digit Recognition",
        "version": "1.0.0",
        "model_loaded": is_model_ready
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict handwritten digit from uploaded image.
    Accepts multipart/form-data or application/json payloads.
    Persists prediction record to SQLite history.
    """
    if predictor is None:
        return jsonify({
            "success": False,
            "error": "Model service is unavailable."
        }), 503

    try:
        image_bytes, input_type = extract_image_bytes(request)
        result = predictor.predict(image_bytes)

        # Persist prediction record in SQLite database
        prediction_id = database.insert_prediction(
            predicted_digit=result["digit"],
            confidence=result["confidence"],
            top_three=result.get("top_three", []),
            probabilities=result["probabilities"],
            input_type=input_type,
        )

        return jsonify({
            "success": True,
            "prediction_id": prediction_id,
            "predicted_digit": result["digit"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "top_three": result.get("top_three", []),
            "preprocessing": result.get("preprocessing", {}),
            "input_type": input_type,
        }), 200

    except (InvalidImageError, UnsupportedFormatError, BlankImageError) as err:
        return jsonify({
            "success": False,
            "error": str(err)
        }), 400

    except RequestEntityTooLarge:
        return jsonify({
            "success": False,
            "error": "Upload exceeds the maximum allowed size of 2MB."
        }), 413

    except Exception as err:
        logger.exception("Unexpected error during digit prediction: %s", err)
        return jsonify({
            "success": False,
            "error": "An error occurred while processing the image. Please try again."
        }), 500


# ===========================================================================
# History, Feedback & Analytics API Routes
# ===========================================================================

@app.route("/api/history", methods=["GET"])
def get_prediction_history():
    """Retrieve history records with optional search, filter, and sorting."""
    digit_param = request.args.get("digit")
    digit = int(digit_param) if digit_param is not None and digit_param.isdigit() else None
    input_type = request.args.get("input_type")
    sort_by = request.args.get("sort_by", "date")
    sort_order = request.args.get("sort_order", "desc")
    search = request.args.get("search")
    limit_param = request.args.get("limit", "50")
    limit = int(limit_param) if limit_param.isdigit() else 50

    records = database.get_history(
        digit=digit,
        input_type=input_type,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
        limit=limit,
    )
    return jsonify({"success": True, "history": records, "count": len(records)}), 200


@app.route("/api/history/<prediction_id>", methods=["GET"])
def get_single_prediction_history(prediction_id):
    """Retrieve single prediction record details."""
    record = database.get_prediction(prediction_id)
    if not record:
        return jsonify({"success": False, "error": f"Record '{prediction_id}' not found."}), 404
    return jsonify({"success": True, "prediction": record}), 200


@app.route("/api/history/<prediction_id>", methods=["DELETE"])
def delete_single_prediction_history(prediction_id):
    """Delete a single prediction record by ID."""
    deleted = database.delete_prediction(prediction_id)
    if not deleted:
        return jsonify({"success": False, "error": f"Record '{prediction_id}' not found."}), 404
    return jsonify({"success": True, "message": f"Record '{prediction_id}' deleted successfully."}), 200


@app.route("/api/history", methods=["DELETE"])
def clear_all_prediction_history():
    """Clear all records from prediction history."""
    count = database.clear_history()
    return jsonify({"success": True, "message": "All prediction history cleared.", "deleted_count": count}), 200


@app.route("/api/history/export", methods=["GET"])
def export_history_csv_endpoint():
    """Export prediction history records as downloadable CSV with formula injection prevention."""
    csv_data = database.generate_history_csv()
    filename = f"mnist_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache",
        },
    )


@app.route("/api/feedback", methods=["POST"])
def submit_prediction_feedback():
    """Save user verification feedback ('correct' or 'incorrect') for a prediction."""
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON payload required."}), 400
    data = request.get_json(silent=True) or {}
    prediction_id = data.get("prediction_id")
    feedback = data.get("feedback")
    actual_digit = data.get("actual_digit")

    if not prediction_id or not isinstance(prediction_id, str):
        return jsonify({"success": False, "error": "Missing or invalid 'prediction_id'."}), 400
    if feedback not in ("correct", "incorrect"):
        return jsonify({"success": False, "error": "'feedback' must be 'correct' or 'incorrect'."}), 400

    if actual_digit is not None:
        try:
            actual_digit = int(actual_digit)
            if not (0 <= actual_digit <= 9):
                return jsonify({"success": False, "error": "'actual_digit' must be between 0 and 9."}), 400
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Invalid 'actual_digit'."}), 400

    try:
        updated = database.update_feedback(prediction_id, feedback, actual_digit)
        if not updated:
            return jsonify({"success": False, "error": f"Record '{prediction_id}' not found."}), 404
        stats = database.get_feedback_stats()
        return jsonify({"success": True, "message": "Feedback recorded successfully.", "stats": stats}), 200
    except Exception as e:
        logger.exception("Error saving feedback: %s", e)
        return jsonify({"success": False, "error": "Failed to save feedback."}), 500


@app.route("/api/feedback/stats", methods=["GET"])
def get_feedback_stats_endpoint():
    """Retrieve feedback correction statistics."""
    stats = database.get_feedback_stats()
    return jsonify({"success": True, "stats": stats}), 200


@app.errorhandler(413)
def handle_large_entity(error):
    """Handle payloads that exceed MAX_CONTENT_LENGTH."""
    return jsonify({
        "success": False,
        "error": "Upload exceeds the maximum allowed size of 2MB."
    }), 413


@app.errorhandler(405)
def handle_method_not_allowed(error):
    """Handle unsupported HTTP methods."""
    return jsonify({
        "success": False,
        "error": "Method not allowed. Use POST for prediction."
    }), 405


@app.errorhandler(404)
def handle_not_found(error):
    """Handle endpoint not found."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def handle_internal_server_error(error):
    """Handle unhandled server exceptions safely without leaking stack traces."""
    logger.exception("Internal Server Error: %s", error)
    return jsonify({
        "success": False,
        "error": "Internal server error. Please try again later."
    }), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)
