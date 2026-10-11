"""
Proof for Exercise 1: this script never imports the training data or calls
fit(). It only loads the saved model file and predicts - run it in a fresh
terminal after train_model.py has created student_pass_model.joblib.
"""

import joblib

model = joblib.load("student_pass_model.joblib")

for hours in (0.5, 4.0, 7.5):
    predicted_class = model.predict([[hours]])[0]
    probability = model.predict_proba([[hours]])[0][1]
    print(f"{hours:>4} study hours -> {'PASS' if predicted_class else 'FAIL'} ({probability:.2f})")
