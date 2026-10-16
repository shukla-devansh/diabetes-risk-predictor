"""Loading, cleaning and splitting the Pima Indians Diabetes data."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

# A living person can't have 0 glucose, blood pressure or BMI, so 0 in these
# columns means "not measured". (0 pregnancies is real, so it's not listed.)
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def zeros_to_nan(X):
    """Mark the fake zeros as NaN so the imputer knows they're missing."""
    X = X.copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X


def build_preprocessor():
    """Zeros -> NaN -> column median.

    The imputer learns its medians in fit() (on training data only) and reuses
    them in transform(), which is what keeps test data from leaking in.
    """
    return Pipeline([
        ("zeros_to_nan", FunctionTransformer(zeros_to_nan, feature_names_out="one-to-one")),
        ("median_impute", SimpleImputer(strategy="median")),
    ]).set_output(transform="pandas")  # keep column names, SHAP needs them later


def split_data(df, test_size=0.2, random_state=42):
    """Stratified split: train and test keep the same ~35% diabetic share."""
    return train_test_split(
        df[FEATURES], df[TARGET],
        test_size=test_size, stratify=df[TARGET], random_state=random_state,
    )


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} rows ({y_train.mean():.1%} diabetic)")
    print(f"Test:  {len(X_test)} rows ({y_test.mean():.1%} diabetic)")

    pre = build_preprocessor().fit(X_train)
    medians = pd.Series(pre.named_steps["median_impute"].statistics_, index=FEATURES)
    print("\nMedians learned from TRAINING data only:")
    print(medians[ZERO_AS_MISSING].to_string())

    before = (X_train[ZERO_AS_MISSING] == 0).sum()
    after = (pre.transform(X_train)[ZERO_AS_MISSING] == 0).sum()
    print("\nZeros before -> after cleaning:")
    print(pd.DataFrame({"before": before, "after": after}).to_string())
