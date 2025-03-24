import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load models
duration_pipe = joblib.load('duration_model.pkl')
recurrence_pipe = joblib.load('recurrence_model.pkl')

st.title("NCR Prediction System")

# Input form
with st.form("ncr_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category", ['Env', 'Safety', 'Quality', 'P&D', 'Risk'])
        package = st.selectbox("Package", ['PKG2A', 'PKG3', 'PKG4', 'PKG5', 'PKG6', 'GENE'])
        contractor = st.selectbox("Contractor", ['APSB', 'SUNCON', 'IJMC', 'ROHAS'])
        
    with col2:
        nature = st.selectbox("Nature of NCR", [
            'Environment Control', 'Safety Control', 'Documentation',
            'Schedule waste management', 'Workmanship'
        ])
        date_issued = st.date_input("Date Issued", datetime.today())
        expected_completion = st.date_input("Expected Completion Date", datetime.today())
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input dataframe
    input_data = pd.DataFrame([{
        'Category': category,
        'Package': package,
        'Contractor': contractor,
        'Nature of NCR': nature,
        'Date Issued': date_issued,
        'Expected Completion Date': expected_completion
    }])
    
    # Calculate days until expected
    input_data['Days_Until_Expected'] = (pd.to_datetime(input_data['Expected Completion Date']) - 
                                       pd.to_datetime(input_data['Date Issued'])).dt.days
    
    # Prepare for prediction
    X = input_data.drop(['Date Issued', 'Expected Completion Date'], axis=1)
    
    # Make predictions
    duration = duration_pipe.predict(X)[0]
    recurrence_prob = recurrence_pipe.predict_proba(X)[0][1]
    
    # Display results
    st.subheader("Prediction Results")
    st.metric("Predicted Closure Duration", f"{round(duration)} days")
    st.metric("Recurrence Probability", f"{round(recurrence_prob*100)}%")
    
    # Interpretation
    st.markdown("""
    **Interpretation Guide:**
    - 🟢 Closure Duration < Expected Days: Likely on-time
    - 🟠 Closure Duration ±10% of Expected: Monitor closely
    - 🔴 Closure Duration > Expected: High risk of delay
    - 🔁 Recurrence Probability > 60%: High chance of recurring issue
    """)

st.markdown("---")
st.write("Developed by Mohd Fadhil & Mohd Taufik - RTS Project Team")