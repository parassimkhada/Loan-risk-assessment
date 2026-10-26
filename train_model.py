"""Train the loan risk assessment model."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
TARGET = "loan_status"


def train(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    expected = NUMERIC + CATEGORICAL + [TARGET]
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    original_rows = len(df)

    # Two clearly invalid 123-year employment histories and five ages above 100.
    valid = df["person_age"].between(18, 100) & (
        df["person_emp_length"].isna() |
        (df["person_emp_length"] <= df["person_age"] - 14)
    )
    df = df.loc[valid].copy()
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    if set(y.unique()) != {0, 1}:
        raise ValueError("loan_status must have both classes 0 and 1")

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    pipeline = Pipeline([
        ("preprocess", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    class_1_index = list(pipeline.named_steps["classifier"].classes_).index(1)
    scores = pipeline.predict_proba(X_test)[:, class_1_index]
    metrics = {
        "original_rows": int(original_rows),
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_1_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "default_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "default_recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "data_currency": "Not specified by dataset",
        "split": "80/20 stratified random split, seed 42",

    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "loan_risk_pipeline.joblib", compress=3)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return pipeline, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    train(args.data, args.output)
