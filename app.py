import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load pre-trained models
duration_model = joblib.load('models/duration_model.pkl')
recurrence_model = joblib.load('models/recurrence_model.pkl')

# Input widgets for user data
st.title("NCR Closure Prediction")
category = st.selectbox("Category", ["Quality", "Safety", "Env", "Risk", "P&D"])
nature_of_ncr = st.text_input("Nature of NCR (e.g., Workmanship)")
package = st.selectbox("Package", ["PKG2A", "PKG3", "PKG4", "PKG5", "PKG6"])
contractor = st.selectbox("Contractor", ["APSB", "SUNCON", "IJMC"])
date_issued = st.date_input("Date Issued")

if st.button("Predict"):
    # Preprocess inputs (mimic training data preprocessing)
    input_data = pd.DataFrame({
        'Category': [category],
        'Nature of NCR': [nature_of_ncr],
        'Package': [package],
        'Contractor': [contractor],
        'Date Issued': [date_issued]
    })

    # One-hot encode categorical variables (ensure alignment with training)
    input_data = pd.get_dummies(input_data).reindex(columns=training_columns, fill_value=0)

    # Predict
    duration_pred = duration_model.predict(input_data)[0]
    recurrence_prob = recurrence_model.predict_proba(input_data)[0][1]

    # Display results
    st.subheader("Predictions")
    st.write(f"Predicted Closure Duration: **{duration_pred:.1f} days**")
    st.write(f"Recurrence Probability: **{recurrence_prob*100:.2f}%**")
