"""
Exercise 2 (retraining step) - the original model only knows study hours,
so attendance cannot influence its predictions. This script trains a NEW
model on two features: study_hours and attendance_percentage.

The data below is synthetic (generated with a fixed random seed so results
are repeatable). In a real project you would load actual student records.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_FILENAME = "student_pass_attendance_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]

rng = np.random.default_rng(42)
n = 300

hours = rng.uniform(0, 10, n)
attendance = rng.uniform(40, 100, n)

# Pass chance rises with both study hours and attendance, plus some noise.
score = 0.7 * hours + 0.06 * attendance + rng.normal(0, 1.0, n)
passed = (score > 6.0).astype(int)

data = pd.DataFrame({"study_hours": hours, "attendance_percentage": attendance, "passed": passed})

X_train, X_test, y_train, y_test = train_test_split(
    data[FEATURES], data["passed"], test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)  # a DataFrame keeps the column names with the model

print(f"Test accuracy: {model.score(X_test, y_test):.2f}")
print("Coefficients:", {f: round(float(c), 3) for f, c in zip(FEATURES, model.coef_[0])})

joblib.dump(model, MODEL_FILENAME)
print(f"Saved model to: {MODEL_FILENAME}")
