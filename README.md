# Diabetes Risk Predictor

[![CI](https://github.com/shukla-devansh/diabetes-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/shukla-devansh/diabetes-risk-predictor/actions/workflows/ci.yml)

A Streamlit app that predicts diabetes risk from 8 basic health measurements and uses SHAP to show which measurements pushed the risk up or down.

![App screenshot](docs/app_screenshot.jpg)

## Problem

Diabetes is often diagnosed late. A model that flags high-risk patients from routine measurements could help decide who needs a follow-up blood test. In healthcare a risk score alone isn't very useful, though: a clinician needs to see why the model flagged someone. That's why every prediction here comes with an explanation.

## Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): 768 women of Pima heritage aged 21+, with 8 features (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age) and an `Outcome` column.

From the [EDA notebook](notebooks/eda.ipynb):
- 35% of patients are diabetic, so the classes are imbalanced.
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are really missing values. Insulin is missing for almost half the rows.
- Glucose has the strongest correlation with the outcome (0.47).

## Approach

- Replace the impossible zeros with the column median. The medians come from the training set only, inside an sklearn pipeline, to avoid data leakage.
- Use an 80/20 stratified train/test split.
- Train Logistic Regression and Random Forest, both with `class_weight="balanced"`.
- Pick the model with the best 5-fold cross-validated recall on the training set.
- Explain predictions with SHAP.

## Results

Test set (154 patients):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.734 | 0.603 | 0.704 | 0.650 | 0.813 |
| Random Forest (chosen) | 0.721 | 0.577 | 0.759 | 0.656 | 0.817 |

I chose Random Forest because of its higher recall. In screening, missing a diabetic patient is worse than a false alarm, which only leads to an extra test. Random Forest caught 41 of the 54 diabetic patients in the test set, compared with 38 for Logistic Regression, even though its accuracy was slightly lower.

### SHAP

| Feature importance | Effect direction |
|---|---|
| ![SHAP bar](reports/figures/shap_global_bar.png) | ![SHAP beeswarm](reports/figures/shap_beeswarm.png) |

Glucose matters most, followed by BMI and Age, which matches the EDA.

One diabetic patient from the test set:

![SHAP waterfall](reports/figures/shap_patient_waterfall.png)

The model gave her only 45%, which makes her a false negative. Her lower BMI and glucose pulled the score down.

## How to run

Python 3.11, on Windows:

```bash
git clone https://github.com/shukla-devansh/diabetes-risk-predictor.git
cd diabetes-risk-predictor
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
pytest
```

To retrain the models and regenerate the charts:

```bash
python -m src.train
python -m src.explain
```

## Project structure

```
app.py               Streamlit app
data/diabetes.csv    dataset
notebooks/eda.ipynb  exploratory analysis
src/data.py          loading, cleaning, train/test split
src/train.py         training and model comparison
src/predict.py       prediction and SHAP explanation
src/explain.py       SHAP charts
models/              saved model and metrics
tests/               pytest tests
```

## What I learned

- The metric you optimize is a design decision. The model with lower accuracy was the better choice for screening.
- Filling missing values before splitting leaks test data into training. Putting the cleaning inside a pipeline fixes that.
- A 0 isn't always a zero. Nobody has a BMI of 0, and checking for values like that showed half the insulin data was missing.
- An unrestricted Random Forest overfits a dataset this small, so I limited the tree depth.
- SHAP makes it possible to check whether a prediction makes sense, instead of just trusting a number.

## Limitations

- Small dataset (768 rows), so the test scores have a wide margin of error.
- The data only covers Pima women aged 21 and older, so the model may be biased and shouldn't be applied to other groups without testing.
- About half of the insulin values are imputed.
- `class_weight="balanced"` pushes the scores up, so a "57%" should be read as above-average risk, not an exact probability. The Low/Medium/High cut-offs (30% and 60%) are my own choice.
- This is a learning project, not a diagnostic tool.
