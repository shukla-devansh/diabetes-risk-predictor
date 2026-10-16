"""Load the saved model, predict one patient's risk and explain it with SHAP.

SHAP gives each feature a share of the prediction: starting from the average
risk (the "base value"), each feature pushes the risk up or down, and the
pushes add up exactly to this patient's predicted risk.
"""
import joblib
import pandas as pd
import shap

from src.data import FEATURES
from src.train import MODEL_PATH

# Screening cut-offs (our choice, not clinical guidance)
LOW_MAX, MEDIUM_MAX = 0.30, 0.60


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def _to_frame(patient):
    """Accept a dict for one patient or a DataFrame of many."""
    if isinstance(patient, dict):
        patient = pd.DataFrame([patient])
    return patient[FEATURES]


def predict_proba(bundle, patient):
    """Probability (0-1) that the patient has diabetes."""
    return float(bundle["pipeline"].predict_proba(_to_frame(patient))[0, 1])


def risk_label(probability):
    if probability < LOW_MAX:
        return "Low"
    if probability < MEDIUM_MAX:
        return "Medium"
    return "High"


def make_explainer(bundle):
    """Model-agnostic SHAP explainer over the RAW inputs.

    We explain the whole pipeline (cleaning + model), so the explanation talks
    about the values the user typed, and it works whichever model won training.
    """
    pipe = bundle["pipeline"]

    def risk(X):
        return pipe.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]

    # algorithm="exact" tries every combination of the 8 features
    return shap.Explainer(risk, bundle["background"], algorithm="exact")


def explain(bundle, patient, explainer=None):
    """SHAP values for patient(s): one row per patient, in probability units."""
    explainer = explainer or make_explainer(bundle)
    return explainer(_to_frame(patient))


if __name__ == "__main__":
    bundle = load_model()
    example = {
        "Pregnancies": 2, "Glucose": 160, "BloodPressure": 72, "SkinThickness": 30,
        "Insulin": 0, "BMI": 36.5, "DiabetesPedigreeFunction": 0.6, "Age": 45,
    }
    p = predict_proba(bundle, example)
    print(f"Model: {bundle['name']}")
    print(f"Risk: {p:.1%} ({risk_label(p)})")

    sv = explain(bundle, example)[0]
    print(f"\nAverage risk (base value): {sv.base_values:.1%}")
    contrib = pd.Series(sv.values, index=FEATURES).sort_values(key=abs, ascending=False)
    for feature, value in contrib.items():
        print(f"  {feature:<26} {value:+.1%}")
    print(f"Base + contributions = {sv.base_values + sv.values.sum():.1%}")
