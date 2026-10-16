import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import build_preprocessor, load_data, split_data

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.joblib"
METRICS_PATH = MODEL_PATH.with_name("metrics.json")


def main():
    X_train, X_test, y_train, y_test = split_data(load_data())

    models = {
        "Logistic Regression": make_pipeline(
            build_preprocessor(), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000)),
        "Random Forest": make_pipeline(
            build_preprocessor(),
            RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42)),
    }

    results = {}
    for name, model in models.items():
        # choose the model using cross-validation on train, keep the test set for the end
        cv_recall = cross_val_score(model, X_train, y_train, cv=5, scoring="recall").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_recall": round(cv_recall, 3),
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred), 3),
            "recall": round(recall_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 3),
        }

    print(pd.DataFrame(results).T)

    # recall matters most: missing a diabetic patient is worse than a false alarm
    best = max(results, key=lambda name: results[name]["cv_recall"])
    print("best model:", best)

    joblib.dump({"name": best, "model": models[best],
                 "background": X_train.sample(100, random_state=42)}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"best_model": best, "results": results}, indent=2))


if __name__ == "__main__":
    main()
