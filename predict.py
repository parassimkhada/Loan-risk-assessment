"""Pure prediction helpers, also used by Streamlit."""
from pathlib import Path
import joblib
import pandas as pd

FIELDS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]


def load_pipeline(folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parent
    # Only load the model file produced by your own training script.
    return joblib.load(folder / "loan_risk_pipeline.joblib")


def predict_applicant(model, applicant):
    missing = set(FIELDS) - set(applicant)
    if missing:
        raise ValueError(f"Missing applicant fields: {sorted(missing)}")
    row = pd.DataFrame([{field: applicant[field] for field in FIELDS}])
    predicted = int(model.predict(row)[0])
    class_1_index = list(model.classes_).index(1)
    score = float(model.predict_proba(row)[0][class_1_index])
    return predicted, score

def illustrative_card_eligibility(predicted, applicant):
    if predicted != 0 or applicant["cb_person_default_on_file"] != "N":
        return "Not eligible"
    income = applicant["person_income"]
    ratio = applicant["loan_percent_income"]
    if ratio >= 0.4:
        return "Not eligible"
    if income >= 133000:
        return "Meets the Platinum Card rules"
    if income >= 40000:
        return "Meets the Standard Card rules"
    return "Not eligible"
