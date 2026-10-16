import pandas as pd
import pytest

from src.predict import explain, load_model, predict_proba, risk_label

HEALTHY = {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
           "Insulin": 0, "BMI": 22.0, "DiabetesPedigreeFunction": 0.2, "Age": 22}
AT_RISK = {"Pregnancies": 8, "Glucose": 190, "BloodPressure": 80, "SkinThickness": 35,
           "Insulin": 300, "BMI": 42.0, "DiabetesPedigreeFunction": 1.2, "Age": 55}
ALL_MISSING = {**HEALTHY, "Glucose": 0, "BloodPressure": 0, "SkinThickness": 0, "Insulin": 0, "BMI": 0}


@pytest.fixture(scope="module")
def bundle():
    return load_model()


def test_model_loads(bundle):
    assert hasattr(bundle["model"], "predict_proba")


@pytest.mark.parametrize("patient", [HEALTHY, AT_RISK, ALL_MISSING])
def test_predict_returns_probability(bundle, patient):
    p = predict_proba(bundle, patient)
    assert 0 <= p <= 1


def test_at_risk_scores_higher(bundle):
    assert predict_proba(bundle, AT_RISK) > predict_proba(bundle, HEALTHY)


def test_risk_labels():
    assert risk_label(0.1) == "Low"
    assert risk_label(0.45) == "Medium"
    assert risk_label(0.9) == "High"


def test_shap_values_add_up(bundle):
    sv = explain(bundle, pd.DataFrame([AT_RISK]))[0]
    assert sv.base_values + sv.values.sum() == pytest.approx(predict_proba(bundle, AT_RISK))
