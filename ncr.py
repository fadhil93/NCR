import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load models and encoder
try:
    duration_pipe = joblib.load('model/duration_model.pkl')
    recurrence_pipe = joblib.load('model/recurrence_model.pkl')
    encoder = joblib.load('model/encoder.pkl')  # Ensure encoder.pkl is available
except FileNotFoundError as e:
    st.error(f"Missing model file: {e.filename}. Ensure all model files exist in the directory.")
    st.stop()

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
    }])

    # Encode categorical features
    input_encoded = encoder.transform(input_data)  # Transform input data using trained encoder
    encoder_features = encoder.get_feature_names_out()
    
    # Convert back to DataFrame
    X_encoded = pd.DataFrame(input_encoded, columns=encoder_features)

    # **Ensure columns match trained model**
    expected_features = duration_pipe.feature_names_in_
    for col in expected_features:
        if col not in X_encoded.columns:
            X_encoded[col] = 0  # Add missing columns with 0 values

    X_encoded = X_encoded[expected_features]  # Reorder to match training order

    # Make predictions
    duration = duration_pipe.predict(X_encoded)[0]
    recurrence_prob = recurrence_pipe.predict_proba(X_encoded)[0][1]

    # Display results
    st.subheader("Prediction Results")
    st.metric("Predicted Closure Duration", f"{round(duration)} days")
    st.metric("Recurrence Probability", f"{round(recurrence_prob*100)}%")

    # Interpretation guide
    st.markdown("""
    **Interpretation Guide:**
    - 🟢 Closure Duration < Expected Days: Likely on-time
    - 🟠 Closure Duration ±10% of Expected: Monitor closely
    - 🔴 Closure Duration > Expected: High risk of delay
    - 🔁 Recurrence Probability > 60%: High chance of recurring issue
    """)

st.markdown("---")
st.write("Developed by Mohd Fadhil & Mohd Taufik - RTS Project Team")
