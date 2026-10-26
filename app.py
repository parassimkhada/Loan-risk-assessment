"""Streamlit loan-risk research demonstration by Paras Simkhada."""
import streamlit as st
from predict import load_pipeline, predict_applicant, illustrative_card_eligibility

st.set_page_config(page_title="Loan Risk Assessment", page_icon="🏦")
st.title("Loan Risk Assessment & Card Eligibility")
st.write("A machine-learning demonstration using historical loan records.")


@st.cache_resource
def get_model():
    return load_pipeline()

try:
    model = get_model()
except Exception:
    st.error("Model file is missing or incompatible. Keep loan_risk_pipeline.joblib beside app.py and use the pinned requirements.txt.")
    st.stop()

with st.form("loan_form"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual income (dataset units)", min_value=1, max_value=10000000, value=50000, step=1000)
        employment = st.number_input("Employment length (years)", min_value=0.0, max_value=70.0, value=5.0, step=0.5)
        ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        intent = st.selectbox("Loan purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        grade = st.selectbox("Loan grade", list("ABCDEFG"))
    with right:
        loan = st.number_input("Loan amount (dataset units)", min_value=1, max_value=1000000, value=10000, step=500)
        interest = st.number_input("Interest rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        history = st.number_input("Credit history (years)", min_value=0, max_value=60, value=5)
        prior_default = st.selectbox("Previous default on file?", ["N", "Y"])
    ratio = loan / income
    st.caption(f"Loan-to-annual-income ratio (automatically calculated): {ratio:.1%}")
    submit = st.form_submit_button("Predict loan risk")

if submit:
    if employment > age - 14:
        st.error("Employment length cannot exceed age minus 14 years.")
        st.stop()
    applicant = {
        "person_age": age, "person_income": income, "person_emp_length": employment,
        "person_home_ownership": ownership, "loan_intent": intent,
        "loan_grade": grade, "loan_amnt": loan, "loan_int_rate": interest,
        "loan_percent_income": ratio, "cb_person_default_on_file": prior_default,
        "cb_person_cred_hist_length": history,
    }
    try:
        prediction, score = predict_applicant(model, applicant)
    except Exception:
        st.error("Prediction failed. Verify that the trained model matches the app.")
        st.stop()
    if prediction == 1:
        st.warning("Model prediction: higher loan non-payment risk (class 1)")
    else:
        st.success("Model prediction: lower loan non-payment risk (class 0)")
    st.metric("Model's non-payment risk score", f"{score:.1%}")
    st.write("**Illustrative card eligibility:** " + illustrative_card_eligibility(prediction, applicant))

st.divider()
