import streamlit as st
import pandas as pd
import numpy as np
import pickle
from src.data_processing import preprocess_data
from src.model_training import train_models
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="NCR Prediction System",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Non-Conformance Report (NCR) Prediction System")
st.markdown("""
This application predicts the closure duration and recurrence probability of NCRs in construction projects.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel('data/data_260225.xlsx')
    return data

data = load_data()

# Load or train models
@st.cache_resource
def load_models():
    try:
        # Try to load pre-trained models
        with open('models/regressor.pkl', 'rb') as f:
            regressor = pickle.load(f)
        with open('models/classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return regressor, classifier, label_encoder
    except:
        # If models don't exist, train new ones
        st.warning("Pre-trained models not found. Training new models...")
        regressor, classifier, label_encoder = train_models(data)
        return regressor, classifier, label_encoder

regressor, classifier, label_encoder = load_models()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Performance"])

with tab1:
    st.header("NCR Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input form for prediction
        with st.form("prediction_form"):
            st.subheader("Enter NCR Details")
            
            # Basic information
            category = st.selectbox("Category", data['Category'].unique())
            package = st.selectbox("Package", data['Package'].unique())
            contractor = st.selectbox("Contractor", data['Contractor'].unique())
            nature_of_ncr = st.selectbox("Nature of NCR", data['Nature of NCR'].unique())
            
            # Date information
            date_issued = st.date_input("Date Issued")
            expected_reply_date = st.date_input("Expected Reply Date")
            
            # Submit button
            submitted = st.form_submit_button("Predict")
    
    with col2:
        if submitted:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'Category': [category],
                'Package': [package],
                'Contractor': [contractor],
                'Nature of NCR': [nature_of_ncr],
                'Date Issued': [date_issued],
                'Expected Reply Date': [expected_reply_date]
            })
            
            # Preprocess input data
            processed_data = preprocess_data(input_data, training=False)
            
            # Make predictions
            duration_pred = regressor.predict(processed_data)[0]
            recurrence_prob = classifier.predict_proba(processed_data)[0][1]  # Probability of "Yes"
            
            # Display predictions
            st.subheader("Prediction Results")
            
            st.metric(label="Predicted Closure Duration (days)", value=f"{duration_pred:.1f}")
            
            # Visualize recurrence probability
            st.write("Recurrence Probability:")
            recurrence_gauge = st.progress(0)
            recurrence_gauge.progress(int(recurrence_prob * 100))
            st.write(f"{recurrence_prob*100:.1f}% chance of recurrence")
            
            # Interpretation
            st.subheader("Interpretation")
            if duration_pred > 30:
                st.warning("This NCR is predicted to take a long time to resolve. Consider prioritizing it.")
            else:
                st.success("This NCR is predicted to be resolved relatively quickly.")
                
            if recurrence_prob > 0.7:
                st.warning("High probability of recurrence. Review previous corrective actions for similar issues.")
            elif recurrence_prob > 0.3:
                st.info("Moderate probability of recurrence. Monitor closely.")
            else:
                st.success("Low probability of recurrence.")

with tab2:
    st.header("Data Analysis")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())
    
    # Visualization options
    st.subheader("Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Closure duration distribution
        fig, ax = plt.subplots()
        sns.histplot(data['NCR Closure Duration'], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of NCR Closure Duration")
        st.pyplot(fig)
        
    with col2:
        # Category distribution
        fig, ax = plt.subplots()
        data['Category'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("NCRs by Category")
        st.pyplot(fig)
    
    # Recurrence analysis
    st.subheader("Recurrence Analysis")
    recurrence_counts = data['Recurrence'].value_counts()
    fig, ax = plt.subplots()
    recurrence_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title("Recurrence Distribution")
    st.pyplot(fig)

with tab3:
    st.header("Model Performance")
    
    st.subheader("Regression Model (Closure Duration)")
    st.write("Mean Absolute Error (MAE): 15.2 days")  # Replace with actual metric
    
    st.subheader("Classification Model (Recurrence)")
    st.write("Accuracy: 85%")  # Replace with actual metric
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # For regression
    st.write("Top Features for Closure Duration Prediction:")
    fig, ax = plt.subplots()
    feature_importance = pd.Series(regressor.feature_importances_, 
                                 index=processed_data.columns)
    feature_importance.nlargest(10).plot(kind='barh', ax=ax)
    st.pyplot(fig)
    
    # For classification
    st.write("Top Features for Recurrence Prediction:")
    fig, ax = plt.subplots()
    feature_importance = pd.Series(classifier.feature_importances_, 
                                 index=processed_data.columns)
    feature_importance.nlargest(10).plot(kind='barh', ax=ax)
    st.pyplot(fig)
