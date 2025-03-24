import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load saved models and encoder
regressor = joblib.load('models/closure_duration_model.joblib')
classifier = joblib.load('models/recurrence_model.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert dates to datetime and calculate features
    date_columns = ['Date Issued', 'Expected Reply Date', 'Expected Completion Date']
    for col in date_columns:
        input_data[col] = pd.to_datetime(input_data[col], format='%d/%m/%Y', errors='coerce')
    
    # Calculate Reply Delay (assuming Actual Reply Date is provided)
    input_data['Reply_Delay'] = (input_data['Actual Reply Date'] - input_data['Expected Reply Date']).dt.days
    
    # One-hot encode categorical variables (ensure same structure as training data)
    categorical_cols = ['Category', 'Status', 'Nature of NCR', 'Package', 'Contractor']
    input_data = pd.get_dummies(input_data, columns=categorical_cols)
    
    # Ensure all required columns are present
    # (Add missing dummy columns with 0s)
    expected_columns = joblib.load('models/expected_columns.joblib')  # Save this during training
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    return input_data[expected_columns]

# Streamlit UI
st.title('NCR Prediction Dashboard')

# Input form
with st.form("ncr_form"):
    st.header("Enter NCR Details")
    
    # Basic info
    category = st.selectbox('Category', ['Safety', 'Quality', 'Environment'])
    nature_of_ncr = st.text_input('Nature of NCR')
    package = st.text_input('Package')
    
    # Date inputs
    date_issued = st.date_input('Date Issued')
    expected_reply = st.date_input('Expected Reply Date')
    expected_completion = st.date_input('Expected Completion Date')
    
    # Submit button
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Create input DataFrame
        input_dict = {
            'Category': [category],
            'Nature of NCR': [nature_of_ncr],
            'Package': [package],
            'Date Issued': [date_issued.strftime('%d/%m/%Y')],
            'Expected Reply Date': [expected_reply.strftime('%d/%m/%Y')],
            'Expected Completion Date': [expected_completion.strftime('%d/%m/%Y')],
            'Status': ['Open']  # Assuming default status
        }
        
        input_df = pd.DataFrame(input_dict)
        processed_df = preprocess_input(input_df)
        
        # Predictions
        duration_pred = regressor.predict(processed_df)[0]
        recurrence_prob = classifier.predict_proba(processed_df)[0][1]
        
        # Display results
        st.subheader("Predictions")
        st.metric("Predicted Closure Duration (days)", f"{duration_pred:.1f}")
        st.metric("Recurrence Probability", f"{recurrence_prob*100:.1f}%")

# Add additional visualizations/features as needed