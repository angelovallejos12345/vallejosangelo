# Student Pass Predictor - Exercises 1-3

## Setup and run order

```
pip install -r requirements.txt

python train_model.py              # Ex 1: trains + saves student_pass_model.joblib, reloads it, 3 predictions
python predict_only.py             # Ex 1: loads the file in a fresh process - no training code at all
python train_model_attendance.py   # Ex 2: retrains with attendance -> student_pass_attendance_model.joblib
python app.py                      # Ex 2: Flask API on http://127.0.0.1:5000
streamlit run streamlit_app.py     # Ex 3: needs student_pass_model.joblib from train_model.py
```

Model files are not shipped on purpose - a `.joblib` file should be created by the
same scikit-learn version that loads it, so run the two training scripts first.

## Exercise 1 - why no retraining is needed

`model.fit()` is the step that *learns*: it finds the numbers stored inside the model
(for logistic regression, a coefficient and an intercept). `joblib.dump()` writes those
learned numbers to `student_pass_model.joblib`, and `joblib.load()` restores them exactly.
`predict()` only plugs a new input into the stored formula, so neither the training data
nor `fit()` is needed again - which is why `predict_only.py` works without importing any
training code, and why a deployed app can load the file once at startup and answer instantly.

## Exercise 2 - testing /predict

The original model only used study hours, so attendance could not affect it. `app.py`
loads the retrained two-feature model instead. Same 3 study hours, different attendance:
40% -> 0.14 pass probability, 70% -> 0.58, 100% -> 0.92.

**curl.exe (PowerShell / cmd).** Quoting JSON inline is fragile on Windows, so put it in a file:

```
# payload.json
{"study_hours": 6, "attendance_percentage": 90}

curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "@payload.json"
```

Inline form (cmd / PowerShell 5.1):

```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 90}"
```

Query-string form (no JSON quoting needed):

```
curl.exe "http://127.0.0.1:5000/predict?study_hours=4&attendance_percentage=70"
```

**Postman:** POST `http://127.0.0.1:5000/predict`, Body -> raw -> JSON, paste the payload.

Expected success response:

```json
{"attendance_percentage": 90.0, "pass_probability": 0.993, "prediction": 1, "result": "pass", "study_hours": 6.0}
```

Validation cases (all return HTTP 400 with an `error` message): attendance 101 or -1,
missing field, non-numeric value, `true`/`false`, negative study hours, body that isn't JSON.
Attendance 0 and 100 are accepted (the range is inclusive).
