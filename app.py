import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import FEATURES
from src.predict import explain, load_model, predict_proba, risk_label

st.set_page_config(page_title="Diabetes Risk Predictor")


@st.cache_resource
def get_model():
    return load_model()


bundle = get_model()

st.title("Diabetes Risk Predictor")
st.write("Enter the patient's measurements to estimate their diabetes risk.")

# label, min, max, default
fields = {
    "Pregnancies": ("Pregnancies", 0, 20, 2),
    "Glucose": ("Glucose (mg/dL)", 0, 250, 120),
    "BloodPressure": ("Blood pressure (mm Hg)", 0, 150, 72),
    "SkinThickness": ("Skin thickness (mm), 0 if unknown", 0, 100, 29),
    "Insulin": ("Insulin (µU/mL), 0 if unknown", 0, 900, 0),
    "BMI": ("BMI", 0.0, 70.0, 32.0),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", 0.0, 2.5, 0.37),
    "Age": ("Age", 21, 100, 33),
}

with st.form("patient"):
    col1, col2 = st.columns(2)
    patient = {}
    for i, name in enumerate(FEATURES):
        label, low, high, default = fields[name]
        col = col1 if i % 2 == 0 else col2
        patient[name] = col.number_input(label, low, high, default)
    submitted = st.form_submit_button("Predict")

if submitted:
    prob = predict_proba(bundle, patient)
    st.metric("Diabetes risk", f"{prob:.0%}")
    st.write(f"Risk level: **{risk_label(prob)}**")

    sv = explain(bundle, pd.DataFrame([patient]))[0]
    effects = pd.Series(sv.values * 100, index=[f"{f} = {patient[f]}" for f in FEATURES])
    effects = effects.sort_values(key=abs)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(effects.index, effects, color=["tab:red" if v > 0 else "tab:blue" for v in effects])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Effect on risk (percentage points)")

    st.subheader("What drove this prediction")
    st.pyplot(fig)
    st.caption("Red pushed the risk up, blue pulled it down (SHAP values).")
