from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent.parent / "data" / "diabetes.csv"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# a 0 in these columns means the value wasn't recorded
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def split_data(df):
    return train_test_split(df[FEATURES], df["Outcome"], test_size=0.2,
                            stratify=df["Outcome"], random_state=42)


def build_preprocessor():
    # medians are learned in fit(), so only from the training data
    imputer = SimpleImputer(missing_values=0, strategy="median")
    pre = ColumnTransformer([("impute", imputer, ZERO_AS_MISSING)],
                            remainder="passthrough", verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")
