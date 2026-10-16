import joblib
import pandas as pd
import shap

from src.data import FEATURES
from src.train import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict_proba(bundle, patient):
    X = pd.DataFrame([patient])[FEATURES]
    return float(bundle["model"].predict_proba(X)[0, 1])


def risk_label(prob):
    if prob < 0.3:
        return "Low"
    if prob < 0.6:
        return "Medium"
    return "High"


def explain(bundle, X):
    # explain the whole pipeline so the SHAP values refer to the raw inputs
    def f(data):
        return bundle["model"].predict_proba(pd.DataFrame(data, columns=FEATURES))[:, 1]

    explainer = shap.Explainer(f, bundle["background"], algorithm="exact")
    return explainer(X[FEATURES])
