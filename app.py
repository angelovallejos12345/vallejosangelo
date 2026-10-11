"""
Exercise 2 - Flask API

POST /predict  (JSON body)   {"study_hours": 5, "attendance_percentage": 85}
GET  /predict  (query string) /predict?study_hours=5&attendance_percentage=85

Returns both inputs plus the model's prediction as JSON.

Run train_model_attendance.py first - the original study-hours-only model
cannot use attendance, so this API loads the retrained two-feature model.
"""

import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).parent / "student_pass_attendance_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]

app = Flask(__name__)

# Loaded ONCE when the server starts, not on every request.
if not MODEL_PATH.exists():
    raise SystemExit(f"Model file not found: {MODEL_PATH.name}. Run train_model_attendance.py first.")
model = joblib.load(MODEL_PATH)


def parse_number(data, field):
    """Return (value, error). Rejects missing, non-numeric, NaN and infinite values."""
    raw = data.get(field)
    if raw is None or raw == "":
        return None, f"'{field}' is required."
    if isinstance(raw, bool):  # in Python True/False are numbers; don't accept them
        return None, f"'{field}' must be a number."
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"'{field}' must be a number."
    if not math.isfinite(value):
        return None, f"'{field}' must be a finite number."
    return value, None


@app.get("/")
def index():
    return jsonify(message="Student pass predictor API", endpoint="/predict", inputs=FEATURES)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # JSON body for POST, query string for GET
    data = request.get_json(silent=True) if request.method == "POST" else request.args
    if not isinstance(data, dict) and request.method == "POST":
        return jsonify(error="Send a JSON object, e.g. {\"study_hours\": 5, \"attendance_percentage\": 85}."), 400

    study_hours, error = parse_number(data, "study_hours")
    if error:
        return jsonify(error=error), 400
    attendance, error = parse_number(data, "attendance_percentage")
    if error:
        return jsonify(error=error), 400

    if study_hours < 0:
        return jsonify(error="'study_hours' cannot be negative."), 400
    if not 0 <= attendance <= 100:  # Exercise 2 validation rule
        return jsonify(error="'attendance_percentage' must be between 0 and 100."), 400

    row = pd.DataFrame([[study_hours, attendance]], columns=FEATURES)
    predicted_class = int(model.predict(row)[0])
    pass_probability = float(model.predict_proba(row)[0][1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=predicted_class,
        result="pass" if predicted_class == 1 else "fail",
        pass_probability=round(pass_probability, 4),
    )


if __name__ == "__main__":
    app.run(debug=True)
