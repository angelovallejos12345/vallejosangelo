"""
Exercise 3 - Streamlit interface

Run with:  streamlit run streamlit_app.py
Requires student_pass_model.joblib (created by train_model.py).
"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    """Load the saved model once and reuse it across reruns (no retraining)."""
    return joblib.load(MODEL_PATH)


st.title("🎓 Student Pass Predictor")
st.write("Enter a student's study hours to see whether the model predicts a pass or a fail.")

if not MODEL_PATH.exists():
    st.error(f"Model file `{MODEL_PATH.name}` not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
student_name = st.text_input("Student name", placeholder="e.g. Maria Santos")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=12.0, value=4.0, step=0.5,
    help="Hours studied (0 - 12).",
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("Predict", type="primary"):
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
        st.stop()

    predicted_class = int(model.predict([[study_hours]])[0])
    pass_probability = float(model.predict_proba([[study_hours]])[0][1])
    name = student_name.strip()

    # Result message
    if predicted_class == 1:
        st.success(f"✅ **{name}** is predicted to **PASS** after studying {study_hours:g} hours.")
    else:
        st.error(f"⚠️ **{name}** is predicted to **FAIL** with {study_hours:g} hours of study.")
    st.metric("Estimated probability of passing", f"{pass_probability:.0%}")

    # Short explanation of how the prediction was generated
    st.subheader("How was this prediction generated?")
    if hasattr(model, "coef_"):
        coef = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])
        st.write(
            "The model is a **logistic regression** trained on past students' study hours "
            "and whether they passed. It was **loaded from a saved file**, not retrained. "
            f"It turns your input into a pass probability with the formula "
            f"`1 / (1 + e^-({coef:.2f} × hours {intercept:+.2f}))`. "
            f"For {study_hours:g} hours that gives **{pass_probability:.0%}**, "
            "and a probability of 50% or more is predicted as *pass*."
        )
    else:
        st.write(
            "The model was loaded from a saved file (not retrained) and compares your "
            "study hours with patterns learned from past students."
        )

    # Bonus: chart comparing study hours with the predicted class
    st.subheader("Study hours vs. predicted class")
    grid = pd.DataFrame({"study_hours": [h / 10 for h in range(0, 121)]})
    grid["pass_probability"] = model.predict_proba(grid[["study_hours"]].values)[:, 1]
    student_point = pd.DataFrame(
        {"study_hours": [study_hours], "pass_probability": [pass_probability], "student": [name]}
    )

    curve = alt.Chart(grid).mark_line().encode(
        x=alt.X("study_hours:Q", title="Study hours"),
        y=alt.Y("pass_probability:Q", title="Probability of passing", scale=alt.Scale(domain=[0, 1])),
    )
    threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 5], color="gray").encode(y="y:Q")
    point = alt.Chart(student_point).mark_point(
        size=250, filled=True, color="green" if predicted_class == 1 else "red"
    ).encode(x="study_hours:Q", y="pass_probability:Q", tooltip=["student", "study_hours", "pass_probability"])

    # Fixed size (no use_container_width / width="stretch") so this runs on both old and new Streamlit.
    st.altair_chart((curve + threshold + point).properties(width=600, height=320))
    st.caption(
        "The curve is the model's pass probability for every study-hours value. "
        "Above the dashed 50% line the predicted class is *pass*; below it, *fail*. "
        "The dot marks this student."
    )
