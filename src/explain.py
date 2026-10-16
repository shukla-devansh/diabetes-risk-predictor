from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data import load_data, split_data
from src.predict import explain, load_model

FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"


def save(name):
    plt.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_model()
    _, X_test, _, y_test = split_data(load_data())
    sv = explain(bundle, X_test)

    shap.plots.bar(sv, show=False)
    save("shap_global_bar.png")

    shap.plots.beeswarm(sv, show=False)
    save("shap_beeswarm.png")

    # first diabetic patient in the test set
    i = list(y_test).index(1)
    shap.plots.waterfall(sv[i], show=False)
    save("shap_patient_waterfall.png")


if __name__ == "__main__":
    main()
