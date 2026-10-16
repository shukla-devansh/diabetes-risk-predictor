"""Streamlit app: enter patient values -> diabetes risk + SHAP explanation.

Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import FEATURES, ZERO_AS_MISSING
from src.predict import explain, load_model, make_explainer, predict_proba, risk_label

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")


# Load once per server, not on every button click
@st.cache_resource
def get_model_and_explainer():
    bundle = load_model()
    return bundle, make_explainer(bundle)


bundle, explainer = get_model_and_explainer()

st.title("🩺 Diabetes Risk Predictor")
st.caption("Predicts diabetes risk from 8 basic health measurements and explains why.")
st.warning(
    "**Not medical advice.** This is a student portfolio project trained on a small "
    "public dataset (768 Pima women). It is not a diagnostic tool. Talk to a doctor "
    "about any health concerns.",
    icon="⚠️",
)

# (label, min, max, default, step)
FIELDS = {
    "Pregnancies": ("Pregnancies", 0, 20, 2, 1),
    "Glucose": ("Glucose (mg/dL, 2-hour OGTT)", 0, 250, 120, 1),
    "BloodPressure": ("Diastolic blood pressure (mm Hg)", 0, 150, 72, 1),
    "SkinThickness": ("Triceps skin fold (mm)", 0, 100, 29, 1),
    "Insulin": ("2-hour serum insulin (µU/mL)", 0, 900, 0, 1),
    "BMI": ("BMI (kg/m²)", 0.0, 70.0, 32.0, 0.1),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function (family history score)", 0.0, 2.5, 0.37, 0.01),
    "Age": ("Age (years)", 21, 100, 33, 1),
}
MISSING_HELP = "Enter 0 if unknown: it's treated as missing and filled with the typical value."

with st.form("patient"):
    cols = st.columns(2)
    patient = {}
    for i, name in enumerate(FEATURES):
        label, lo, hi, default, step = FIELDS[name]
        help_text = MISSING_HELP if name in ZERO_AS_MISSING else None
        patient[name] = cols[i % 2].number_input(label, lo, hi, default, step, help=help_text)
    submitted = st.form_submit_button("Predict risk", type="primary")

if submitted:
    prob = predict_proba(bundle, patient)
    label = risk_label(prob)
    colour = {"Low": "green", "Medium": "orange", "High": "red"}[label]

    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted risk", f"{prob:.0%}")
    c2.markdown(f"### :{colour}[{label} risk]")
    st.caption("Low < 30% ≤ Medium < 60% ≤ High. These cut-offs are illustrative, not clinical.")

    st.subheader("Why this prediction?")
    sv = explain(bundle, patient, explainer)[0]
    shown = {f: "unknown" if f in ZERO_AS_MISSING and patient[f] == 0 else patient[f] for f in FEATURES}
    contrib = (
        pd.Series(sv.values, index=[f"{f} = {shown[f]}" for f in FEATURES])
        .sort_values(key=abs)  # biggest effect ends up at the top of the chart
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(contrib.index, contrib.values * 100,
            color=["#d62728" if v > 0 else "#1f77b4" for v in contrib.values])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Change in risk (percentage points)")
    ax.set_title(f"Average patient risk {sv.base_values:.0%} → this patient {prob:.0%}")
    st.pyplot(fig)
    st.caption(
        "🔴 Red bars pushed the risk **up**, 🔵 blue bars pulled it **down**. "
        "Starting from the average risk, the bars add up to this patient's prediction (SHAP values)."
    )

    top = contrib.abs().idxmax().split(" = ")[0]
    st.info(f"The biggest factor for this patient was **{top}**.")

st.divider()
st.caption(f"Model: {bundle['name']} · Data: Pima Indians Diabetes dataset · Explanations: SHAP")
