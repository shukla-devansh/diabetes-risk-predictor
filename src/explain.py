"""Save SHAP charts: global feature importance + one patient's explanation.

Global importance = the average size of each feature's SHAP push across all
test patients, i.e. "which features does the model lean on most overall?".
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw straight to files, no window needed
import matplotlib.pyplot as plt
import shap

from src.data import load_data, split_data
from src.predict import explain, load_model, make_explainer

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved reports/figures/{name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_model()
    _, X_test, _, y_test = split_data(load_data())

    sv = explain(bundle, X_test, make_explainer(bundle))

    # Bar: mean |SHAP| per feature = overall importance
    shap.plots.bar(sv, show=False)
    plt.title("Global feature importance (mean |SHAP|, test set)")
    save("shap_global_bar.png")

    # Beeswarm: one dot per patient, colour = feature value (red = high).
    # Shows direction too, e.g. high glucose -> pushes risk up.
    shap.plots.beeswarm(sv, show=False)
    plt.title("How each feature pushes risk (test set)")
    save("shap_beeswarm.png")

    # Waterfall for one real diabetic test patient, step by step from base risk
    idx = list(y_test).index(1)
    shap.plots.waterfall(sv[idx], show=False)
    plt.title("Why the model scored this patient")
    save("shap_patient_waterfall.png")


if __name__ == "__main__":
    main()
