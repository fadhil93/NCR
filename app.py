import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load pre-trained models
regressor = joblib.load('models/regressor.pkl')
classifier = joblib.load('models/classifier.pkl')

# Feature engineering function
def preprocess_input(input_df):
    # Date calculations
    input_df['NCR Closure Duration'] = (input_df['Closed Date'] - input_df['Date Issued']).dt.days
    input_df['Reply_Delay'] = (input_df['Actual Reply Date'] - input_df['Expected Reply Date']).dt.days
    
    # Categorical encoding (match training data columns)
    categorical_cols = ['Category', 'Status', 'Nature of NCR', 'Package', 'Contractor']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Add missing columns present in training
    expected_columns = [
        'Category_Quality', 'Status_Closed', 'Nature of NCR_Workmanship', 
        'Package_PKG2A', 'Contractor_APSB'
    ]  # Add all your actual columns here
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    return input_df

# Streamlit UI
st.title('NCR Prediction System')

# Input form
with st.form('ncr_form'):
    category = st.selectbox('Category', ['Quality', 'Safety', 'Env', 'Risk', 'P&D'])
    nature = st.text_input('Nature of NCR')
    package = st.selectbox('Package', ['PKG2A', 'PKG3', 'PKG4', 'PKG5', 'PKG6'])
    
    date_issued = st.date_input('Date Issued')
    expected_reply = st.date_input('Expected Reply Date')
    actual_reply = st.date_input('Actual Reply Date')
    
    submit = st.form_submit_button('Predict')

if submit:
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Category': category,
        'Nature of NCR': nature,
        'Package': package,
        'Date Issued': date_issued,
        'Expected Reply Date': expected_reply,
        'Actual Reply Date': actual_reply
    }])
    
    # Preprocess input
    processed_data = preprocess_input(input_data)
    
    # Predict
    duration_pred = regressor.predict(processed_data)
    recurrence_pred = classifier.predict(processed_data)
    
    # Display results
    st.subheader('Predictions')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Closure Days", f"{duration_pred[0]:.1f}")
    with col2:
        st.metric("Recurrence Risk", "High" if recurrence_pred[0] == 1 else "Low")