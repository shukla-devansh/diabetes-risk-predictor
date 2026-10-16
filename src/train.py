"""Train Logistic Regression and Random Forest, compare them, save the best.

Why recall matters most here: recall = "of all patients who really have
diabetes, how many did we catch?". In screening, a missed case (false negative)
goes home untreated, while a false alarm only costs a follow-up blood test.
So we pick the model that misses the fewest real cases.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import build_preprocessor, load_data, split_data

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def build_models():
    # class_weight="balanced" makes each missed diabetic count ~2x more during
    # training (there are ~2x fewer of them), nudging the model toward recall.
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor()),
            # LR compares coefficients across features, so put them on one scale
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", build_preprocessor()),
            # max_depth / min_samples_leaf stop trees from memorising 614 rows
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=5,
                class_weight="balanced", random_state=42,
            )),
        ]),
    }


def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    X_train, X_test, y_train, y_test = split_data(load_data())

    results = {}
    for name, pipe in build_models().items():
        # Choose the winner with 5-fold cross-validation on the TRAINING set, so
        # the test set stays an untouched final exam.
        cv_recall = cross_val_score(pipe, X_train, y_train, cv=5, scoring="recall").mean()
        pipe.fit(X_train, y_train)
        results[name] = {
            "pipeline": pipe,
            "metrics": {"cv_recall": cv_recall, **evaluate(pipe, X_test, y_test)},
            # A big gap between train and test accuracy is a sign of overfitting
            "train_accuracy": accuracy_score(y_train, pipe.predict(X_train)),
        }

    table = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T.round(3)
    print("Test-set results (cv_recall = 5-fold CV on training data):")
    print(table.to_string())
    for name, r in results.items():
        print(f"{name}: train accuracy {r['train_accuracy']:.3f} vs test {r['metrics']['accuracy']:.3f}")

    best_name = max(results, key=lambda n: results[n]["metrics"]["cv_recall"])
    print(f"\nBest by cross-validated recall: {best_name}")

    MODELS_DIR.mkdir(exist_ok=True)
    # Save the whole pipeline (cleaning + scaling + model) so predict.py can
    # feed raw patient values in, plus a training sample for SHAP to compare against.
    joblib.dump({
        "name": best_name,
        "pipeline": results[best_name]["pipeline"],
        "background": X_train.sample(100, random_state=42),
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(
        {"best_model": best_name, "results": table.to_dict(orient="index")}, indent=2,
    ))
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name} to models/")


if __name__ == "__main__":
    main()
