"""
Exercise 1 - Model Persistence

Trains a simple "will the student pass?" classifier from study hours,
saves it with joblib.dump() under a NEW filename, then loads it back with
joblib.load() and makes three predictions without retraining.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Changed from the original filename. Everything that loads the model
# (app.py, streamlit_app.py, predict_only.py) must use this same name.
MODEL_FILENAME = "student_pass_model.joblib"

# ---------------------------------------------------------------------------
# 1. Training data: hours studied -> passed (1) or failed (0)
#    (a few borderline students around 3-4 hours so the model is not a
#    perfectly clean cut-off)
# ---------------------------------------------------------------------------
study_hours = np.array(
    [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
).reshape(-1, 1)  # sklearn expects a 2-D array: one row per student
passed = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1])

# ---------------------------------------------------------------------------
# 2. Train (this is the slow/expensive step - done once)
# ---------------------------------------------------------------------------
model = LogisticRegression()
model.fit(study_hours, passed)
print(f"Trained. Accuracy on training data: {model.score(study_hours, passed):.2f}")

# ---------------------------------------------------------------------------
# 3. Save the trained model to disk with joblib.dump()
# ---------------------------------------------------------------------------
joblib.dump(model, MODEL_FILENAME)
print(f"Saved model to: {MODEL_FILENAME}")

# ---------------------------------------------------------------------------
# 4. Load it back with joblib.load() into a brand-new variable
# ---------------------------------------------------------------------------
loaded_model = joblib.load(MODEL_FILENAME)
print(f"Loaded model from: {MODEL_FILENAME}\n")

# ---------------------------------------------------------------------------
# 5. Three predictions with different study hours - using the LOADED model
# ---------------------------------------------------------------------------
for hours in (1.0, 3.5, 6.0):
    predicted_class = loaded_model.predict([[hours]])[0]
    pass_probability = loaded_model.predict_proba([[hours]])[0][1]
    label = "PASS" if predicted_class == 1 else "FAIL"
    print(f"Study hours = {hours:>4}  ->  {label}  (probability of passing: {pass_probability:.2f})")

# ---------------------------------------------------------------------------
# 6. Why no retraining was needed
# ---------------------------------------------------------------------------
print(
    "\nWhy no retraining? Training (model.fit) is what *learns* the numbers\n"
    "stored inside the model - for logistic regression, one coefficient and\n"
    f"one intercept: coef={loaded_model.coef_[0][0]:.3f}, "
    f"intercept={loaded_model.intercept_[0]:.3f}.\n"
    "joblib.dump() writes those learned numbers to the file, and joblib.load()\n"
    "restores them exactly. predict() only plugs a new input into the stored\n"
    "formula, so the original training data and fit() are not needed again."
)
