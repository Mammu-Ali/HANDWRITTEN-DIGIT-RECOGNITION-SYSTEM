"""
Database persistence module for Handwritten Digit Recognition.
Manages persistent prediction history, user verification feedback,
filtering, search, and CSV export using SQLite.
"""

import csv
import io
import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Establish a thread-safe connection to SQLite database."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize database tables and indexes if they do not exist."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                predicted_digit INTEGER NOT NULL,
                confidence REAL NOT NULL,
                top_three TEXT NOT NULL,
                probabilities TEXT NOT NULL,
                input_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                feedback TEXT DEFAULT NULL,
                actual_digit INTEGER DEFAULT NULL,
                created_at REAL NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_digit ON predictions(predicted_digit)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_type ON predictions(input_type)")
        conn.commit()
    finally:
        conn.close()


def insert_prediction(
    predicted_digit: int,
    confidence: float,
    top_three: List[Dict[str, Any]],
    probabilities: Dict[str, float],
    input_type: str = "Canvas",
    prediction_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Insert a new prediction record into SQLite."""
    init_db(db_path)
    now = time.time()
    if not prediction_id:
        prediction_id = f"pred_{int(now * 1000)}_{uuid.uuid4().hex[:6]}"
    if not timestamp:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                id, predicted_digit, confidence, top_three, probabilities,
                input_type, timestamp, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                int(predicted_digit),
                float(confidence),
                json.dumps(top_three),
                json.dumps(probabilities),
                input_type,
                timestamp,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return prediction_id


def get_history(
    digit: Optional[int] = None,
    input_type: Optional[str] = None,
    sort_by: str = "date",
    sort_order: str = "desc",
    search: Optional[str] = None,
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Query persistent prediction records with flexible search, filter, and sort options."""
    init_db(db_path)
    query = "SELECT * FROM predictions WHERE 1=1"
    params: List[Any] = []

    if digit is not None and 0 <= digit <= 9:
        query += " AND predicted_digit = ?"
        params.append(digit)

    if input_type and input_type in ("Canvas", "Upload", "Sample"):
        query += " AND input_type = ?"
        params.append(input_type)

    if search and search.strip():
        term = f"%{search.strip()}%"
        query += " AND (id LIKE ? OR timestamp LIKE ? OR input_type LIKE ?)"
        params.extend([term, term, term])

    # Validate sort parameters
    order = "ASC" if sort_order.lower() == "asc" else "DESC"
    if sort_by == "confidence":
        query += f" ORDER BY confidence {order}, created_at DESC"
    else:
        query += f" ORDER BY created_at {order}"

    query += " LIMIT ?"
    params.append(max(1, min(200, limit)))

    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "predicted_digit": r["predicted_digit"],
                "confidence": r["confidence"],
                "top_three": json.loads(r["top_three"]),
                "probabilities": json.loads(r["probabilities"]),
                "input_type": r["input_type"],
                "timestamp": r["timestamp"],
                "feedback": r["feedback"],
                "actual_digit": r["actual_digit"],
                "created_at": r["created_at"],
            })
        return results
    finally:
        conn.close()


def get_prediction(prediction_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve an individual prediction record by ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "predicted_digit": row["predicted_digit"],
            "confidence": row["confidence"],
            "top_three": json.loads(row["top_three"]),
            "probabilities": json.loads(row["probabilities"]),
            "input_type": row["input_type"],
            "timestamp": row["timestamp"],
            "feedback": row["feedback"],
            "actual_digit": row["actual_digit"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def delete_prediction(prediction_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete a single prediction record by ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def clear_history(db_path: str = DEFAULT_DB_PATH) -> int:
    """Delete all prediction records from SQLite database."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def update_feedback(
    prediction_id: str,
    feedback: str,
    actual_digit: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Record user feedback ('correct' or 'incorrect') for a prediction."""
    init_db(db_path)
    if feedback not in ("correct", "incorrect"):
        raise ValueError("Feedback must be 'correct' or 'incorrect'.")
    if actual_digit is not None and not (0 <= actual_digit <= 9):
        raise ValueError("Actual digit must be between 0 and 9.")

    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE predictions
            SET feedback = ?, actual_digit = ?
            WHERE id = ?
            """,
            (feedback, actual_digit, prediction_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_feedback_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Calculate observed correction statistics from user feedback."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback IS NOT NULL")
        total_feedback = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback = 'correct'")
        correct_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback = 'incorrect'")
        incorrect_count = cursor.fetchone()[0]

        observed_accuracy = (
            round((correct_count / total_feedback) * 100, 2)
            if total_feedback > 0
            else None
        )

        return {
            "total_predictions": total_predictions,
            "total_feedback": total_feedback,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "observed_accuracy": observed_accuracy,
        }
    finally:
        conn.close()


def generate_history_csv(db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Generate CSV content from all stored history records.
    Protects against CSV Formula Injection by prepending dangerous characters with a single quote.
    """
    records = get_history(limit=500, db_path=db_path)
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # Header Row
    writer.writerow([
        "Record ID",
        "Timestamp",
        "Input Mode",
        "Predicted Digit",
        "Confidence (%)",
        "Top 2nd Digit",
        "Top 2nd Confidence (%)",
        "Top 3rd Digit",
        "Top 3rd Confidence (%)",
        "Feedback",
        "Actual Digit"
    ])

    def sanitize(val: Any) -> Any:
        if val is None:
            return ""
        s = str(val)
        # Prevent spreadsheet formula injection (=, +, -, @, \t, \r)
        if s.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + s
        return s

    for r in records:
        top_two = r["top_three"][1] if len(r["top_three"]) > 1 else None
        top_three = r["top_three"][2] if len(r["top_three"]) > 2 else None

        writer.writerow([
            sanitize(r["id"]),
            sanitize(r["timestamp"]),
            sanitize(r["input_type"]),
            r["predicted_digit"],
            f"{r['confidence'] * 100:.2f}",
            top_two["digit"] if top_two else "N/A",
            f"{top_two['probability'] * 100:.2f}" if top_two else "N/A",
            top_three["digit"] if top_three else "N/A",
            f"{top_three['probability'] * 100:.2f}" if top_three else "N/A",
            sanitize(r["feedback"] or "unreviewed"),
            r["actual_digit"] if r["actual_digit"] is not None else "N/A"
        ])

    return "\uFEFF" + output.getvalue()
