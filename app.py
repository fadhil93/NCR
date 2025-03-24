import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os

# Set page config
st.set_page_config(page_title="NCR Prediction System", layout="wide")

# Title
st.title("🏗️ NCR Prediction System")
st.markdown("Predicting Timely Closure and Effectiveness of Non-Conformance Reports in Construction Projects")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This application predicts NCR closure duration and recurrence probability.")

# Main content
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Single NCR Prediction")
    with st.form("single_pred_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox("Category", ["Safety", "Quality", "Environment"])
            status = st.selectbox("Status", ["Open", "Closed"])
            nature_ncr = st.text_input("Nature of NCR", "Structural Defect")
            
        with col2:
            package = st.text_input("Package", "MRT-01")
            respond_period = st.number_input("Respond Period (days)", min_value=0, value=5)
            ncr_aging = st.number_input("NCR Aging (days)", min_value=0, value=10)
            
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Mock prediction (replace with your actual model)
            closure_duration = max(0, np.random.normal(30, 10))
            recurrence_prob = np.random.uniform(0, 1)
            
            st.success("Prediction completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Closure Duration", f"{closure_duration:.1f} days")
            with col2:
                st.metric("Recurrence Probability", f"{recurrence_prob*100:.1f}%")
            
            # Interpretation
            if recurrence_prob > 0.7:
                st.error("High recurrence risk - Immediate action recommended")
            elif recurrence_prob > 0.4:
                st.warning("Moderate recurrence risk - Monitor closely")
            else:
                st.success("Low recurrence risk")

with tab2:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload NCR Data (Excel)", type=['xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("Predict for All NCRs"):
            # Add your batch prediction logic here
            progress_bar = st.progress(0)
            
            # Simulate processing
            for i in range(100):
                progress_bar.progress(i + 1)
            
            st.success("Batch prediction completed!")
            st.download_button(
                label="Download Predictions",
                data=df.to_csv().encode('utf-8'),
                file_name='ncr_predictions.csv',
                mime='text/csv'
            )

# Footer
st.markdown("---")
st.markdown("Developed by Mohd Fadhil Bin Mohd Naser & Mohd Taufik Bin Abd Wahid")
