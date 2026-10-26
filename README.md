# Loan Risk Assessment & Card Eligibility

## Local run

Use Python with the pinned scikit-learn version (1.8.0). From this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain using the original dataset

```bash
python train_model.py --data "/path/to/loan risk assessment_dataset (2).csv"
```

This overwrites `loan_risk_pipeline.joblib` and `metrics.json` in the same folder. Only load model files you created or trust; serialized Python models can execute code when loaded.

## Data and evaluation

The original CSV has 32,581 records, 12 columns, and 7,108 class-1 rows (21.82%). Missing values: 895 in `person_emp_length`, 3,116 in `loan_int_rate`. Seven records are excluded for obviously implausible age/employment entries (five ages >100 and two employment lengths of 123 years). No records are excluded based on the target class.

See `metrics.json` for actual test-set performance of the newly trained model. The old reported 93.24% accuracy belongs to a different training run and should not be attributed to this new model.

## Deployment

Upload `app.py`, `predict.py`, `loan_risk_pipeline.joblib`, `requirements.txt` to a GitHub repository and point Streamlit Community Cloud to `app.py`. `train_model.py` and `metrics.json` are optional for deployment. Do not upload applicant data. The original CSV is not needed at runtime.
