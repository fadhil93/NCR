import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load or train models
@st.cache_resource
def load_models():
    # Check if saved models exist
    if os.path.exists('models/regressor.pkl') and os.path.exists('models/classifier.pkl'):
        regressor = pickle.load(open('models/regressor.pkl', 'rb'))
        classifier = pickle.load(open('models/classifier.pkl', 'rb'))
    else:
        # Train models (your existing training code)
        regressor, classifier = train_models()
        # Save models
        os.makedirs('models', exist_ok=True)
        pickle.dump(regressor, open('models/regressor.pkl', 'wb'))
        pickle.dump(classifier, open('models/classifier.pkl', 'wb'))
    return regressor, classifier

def train_models():
    # This should contain your existing model training code from ncr_prediction.py
    # Return trained regressor and classifier
    pass

def main():
    st.title("NCR Prediction System")
    st.write("Predict NCR Closure Duration and Recurrence Probability")
    
    # Load models
    regressor, classifier = load_models()
    
    # Create input form
    with st.form("ncr_input_form"):
        st.header("NCR Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox("Category", ["Safety", "Quality", "Environment"])
            status = st.selectbox("Status", ["Open", "Closed"])
            nature_of_ncr = st.text_input("Nature of NCR")
            package = st.text_input("Package")
            
        with col2:
            respond_period = st.number_input("Respond Period (days)", min_value=0)
            ncr_aging = st.number_input("NCR Aging from Expected Completion date", min_value=0)
            reply_delay = st.number_input("Reply Delay (days)", min_value=0)
            contractor = st.text_input("Contractor")
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Prepare input data
        input_data = {
            'Category': category,
            'Status': status,
            'Nature of NCR': nature_of_ncr,
            'Package': package,
            'Respond Period': respond_period,
            'NCR Aging from Expected Completion date': ncr_aging,
            'Reply_Delay': reply_delay,
            'Contractor': contractor
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess (similar to your training preprocessing)
        # One-hot encode, etc.
        
        # Make predictions
        duration_pred = regressor.predict(input_df)
        recurrence_prob = classifier.predict_proba(input_df)[:, 1][0]
        
        # Display results
        st.subheader("Prediction Results")
        st.metric("Predicted Closure Duration (days)", f"{duration_pred[0]:.1f}")
        st.metric("Recurrence Probability", f"{recurrence_prob*100:.1f}%")
        
        # Interpretation
        if recurrence_prob > 0.7:
            st.warning("High probability of recurrence - consider reviewing previous corrective actions")
        elif recurrence_prob > 0.4:
            st.info("Moderate probability of recurrence - monitor closely")
        else:
            st.success("Low probability of recurrence")

if __name__ == "__main__":
    main()
