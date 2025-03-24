import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime

# Set page config
st.set_page_config(page_title="NCR Prediction System", layout="wide")

# Title
st.title("Non-Conformance Report (NCR) Prediction System")
st.markdown("""
This application predicts:
1. **NCR Closure Duration** (how many days it will take to close an NCR)
2. **Recurrence Probability** (whether an NCR is likely to recur)
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Page", 
                          ["Data Overview", "NCR Duration Prediction", "Recurrence Prediction", "Model Performance"])

# Load data function
@st.cache_data
def load_data():
    data = pd.read_excel('data 260225.xlsx')
    
    # Data cleaning and preprocessing
    data = data.drop(['No.', 'Reference No.', 'Description of Non-Conformance'], axis=1)
    
    # Convert date columns
    date_columns = ['Date Issued', 'Expected Reply Date', 'Actual Reply Date', 
                    'Expected Completion Date', 'Closed Date']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Calculate NCR Closure Duration
    data['NCR Closure Duration'] = (data['Closed Date'] - data['Date Issued']).dt.days
    
    # Calculate Reply Delay
    data['Reply_Delay'] = (data['Actual Reply Date'] - data['Expected Reply Date']).dt.days
    
    # Handle missing values
    data['NCR Closure Duration'] = data['NCR Closure Duration'].fillna(-1)
    
    # Create Recurrence column
    data['Recurrence'] = data.duplicated(subset=['Nature of NCR', 'Package'], keep=False)
    data['Recurrence'] = data['Recurrence'].map({True: 'Yes', False: 'No'})
    
    # Calculate Recurrence Rate
    recurrence_rate = data.groupby(['Nature of NCR', 'Package'])['Recurrence'].apply(
        lambda x: (x == 'Yes').mean()).reset_index()
    recurrence_rate.rename(columns={'Recurrence': 'Recurrence_Rate'}, inplace=True)
    data = data.merge(recurrence_rate, on=['Nature of NCR', 'Package'], how='left')
    
    # Drop rows with missing values
    data.dropna(subset=['Nature of NCR', 'Package'], inplace=True)
    
    return data

# Load data
data = load_data()

# Train models function
@st.cache_resource
def train_models():
    # Prepare data for modeling
    df = data.copy()
    
    # One-hot encode categorical columns
    categorical_cols = ['Category', 'Status', 'Nature of NCR', 'Package', 'Contractor']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Separate features and targets
    X = df.drop(['NCR Closure Duration', 'Recurrence', 
                 'Respond Period', 'NCR Aging from Expected Completion date',
                 'Closed NCR Effectiveness Verification Status',
                 'Date Issued', 'Expected Reply Date', 
                 'Actual Reply Date', 'Expected Completion Date', 'Closed Date'], 
                axis=1, errors='ignore')
    
    y_duration = df['NCR Closure Duration']
    y_recurrence = df['Recurrence']
    
    # Encode Recurrence target column
    label_encoder = LabelEncoder()
    y_recurrence = label_encoder.fit_transform(y_recurrence)
    
    # Split data
    X_train, X_test, y_train_duration, y_test_duration = train_test_split(
        X, y_duration, test_size=0.2, random_state=42)
    _, _, y_train_recurrence, y_test_recurrence = train_test_split(
        X, y_recurrence, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train_duration)
    
    # Train Random Forest Classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train_recurrence)
    
    # Train Linear Regression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train_duration)
    
    # Train SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train_duration)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_test_duration': y_test_duration,
        'y_test_recurrence': y_test_recurrence,
        'regressor': regressor,
        'classifier': classifier,
        'linear_reg': linear_reg,
        'svr': svr,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'features': X.columns
    }

# Train models
models = train_models()

# Data Overview Page
if options == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("Sample Data")
    st.dataframe(data.head())
    
    st.subheader("Data Statistics")
    st.write(data.describe())
    
    st.subheader("NCR Closure Duration Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['NCR Closure Duration'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Recurrence Rate by Category")
    recurrence_by_category = data.groupby('Category')['Recurrence'].value_counts(normalize=True).unstack()
    fig, ax = plt.subplots()
    recurrence_by_category.plot(kind='bar', stacked=True, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# NCR Duration Prediction Page
elif options == "NCR Duration Prediction":
    st.header("Predict NCR Closure Duration")
    
    st.subheader("Enter NCR Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category", data['Category'].unique())
        status = st.selectbox("Status", data['Status'].unique())
        package = st.selectbox("Package", data['Package'].unique())
    
    with col2:
        nature = st.selectbox("Nature of NCR", data['Nature of NCR'].unique())
        contractor = st.selectbox("Contractor", data['Contractor'].unique())
        reply_delay = st.number_input("Reply Delay (days)", min_value=0, max_value=365, value=7)
    
    if st.button("Predict Closure Duration"):
        # Create input dataframe
        input_data = {
            'Reply_Delay': [reply_delay],
            'Recurrence_Rate': [data[data['Nature of NCR'] == nature]['Recurrence_Rate'].mean()]
        }
        
        # Add one-hot encoded features
        for feature in models['features']:
            if feature.startswith('Category_'):
                input_data[feature] = [1 if feature == f'Category_{category}' else 0]
            elif feature.startswith('Status_'):
                input_data[feature] = [1 if feature == f'Status_{status}' else 0]
            elif feature.startswith('Nature of NCR_'):
                input_data[feature] = [1 if feature == f'Nature of NCR_{nature}' else 0]
            elif feature.startswith('Package_'):
                input_data[feature] = [1 if feature == f'Package_{package}' else 0]
            elif feature.startswith('Contractor_'):
                input_data[feature] = [1 if feature == f'Contractor_{contractor}' else 0]
            elif feature not in input_data:
                input_data[feature] = [0]
        
        input_df = pd.DataFrame(input_data)
        
        # Ensure columns match training data
        missing_cols = set(models['X_train'].columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[models['X_train'].columns]
        
        # Make predictions
        rf_pred = models['regressor'].predict(input_df)[0]
        lr_pred = models['linear_reg'].predict(input_df)[0]
        svr_pred = models['svr'].predict(models['scaler'].transform(input_df))[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Random Forest Prediction", f"{int(rf_pred)} days")
        with col2:
            st.metric("Linear Regression Prediction", f"{int(lr_pred)} days")
        with col3:
            st.metric("SVR Prediction", f"{int(svr_pred)} days")
        
        # Feature importance
        st.subheader("Feature Importance (Random Forest)")
        feature_importance = pd.Series(
            models['regressor'].feature_importances_, 
            index=models['X_train'].columns
        )
        top_features = feature_importance.nlargest(10)
        
        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax)
        plt.title('Top 10 Important Features for Duration Prediction')
        st.pyplot(fig)

# Recurrence Prediction Page
elif options == "Recurrence Prediction":
    st.header("Predict NCR Recurrence Probability")
    
    st.subheader("Enter NCR Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category", data['Category'].unique())
        package = st.selectbox("Package", data['Package'].unique())
    
    with col2:
        nature = st.selectbox("Nature of NCR", data['Nature of NCR'].unique())
        reply_delay = st.number_input("Reply Delay (days)", min_value=0, max_value=365, value=7)
    
    if st.button("Predict Recurrence Probability"):
        # Create input dataframe
        input_data = {
            'Reply_Delay': [reply_delay],
            'Recurrence_Rate': [data[data['Nature of NCR'] == nature]['Recurrence_Rate'].mean()]
        }
        
        # Add one-hot encoded features
        for feature in models['features']:
            if feature.startswith('Category_'):
                input_data[feature] = [1 if feature == f'Category_{category}' else 0]
            elif feature.startswith('Nature of NCR_'):
                input_data[feature] = [1 if feature == f'Nature of NCR_{nature}' else 0]
            elif feature.startswith('Package_'):
                input_data[feature] = [1 if feature == f'Package_{package}' else 0]
            elif feature not in input_data:
                input_data[feature] = [0]
        
        input_df = pd.DataFrame(input_data)
        
        # Ensure columns match training data
        missing_cols = set(models['X_train'].columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[models['X_train'].columns]
        
        # Make prediction
        proba = models['classifier'].predict_proba(input_df)[0][1]
        prediction = models['classifier'].predict(input_df)[0]
        prediction_label = models['label_encoder'].inverse_transform([prediction])[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        st.metric("Recurrence Probability", f"{proba*100:.1f}%")
        st.metric("Predicted Recurrence", prediction_label)
        
        # Interpretation
        if proba > 0.7:
            st.warning("High probability of recurrence. Consider reviewing previous corrective actions.")
        elif proba > 0.3:
            st.info("Moderate probability of recurrence. Monitor closely.")
        else:
            st.success("Low probability of recurrence.")

# Model Performance Page
elif options == "Model Performance":
    st.header("Model Performance Evaluation")
    
    st.subheader("NCR Closure Duration Prediction Performance")
    
    # Regression metrics
    y_pred_duration = models['regressor'].predict(models['X_test'])
    mae_rf = mean_absolute_error(models['y_test_duration'], y_pred_duration)
    
    y_pred_duration_lr = models['linear_reg'].predict(models['X_test'])
    mae_lr = mean_absolute_error(models['y_test_duration'], y_pred_duration_lr)
    
    y_pred_duration_svr = models['svr'].predict(models['scaler'].transform(models['X_test']))
    mae_svr = mean_absolute_error(models['y_test_duration'], y_pred_duration_svr)
    
    st.write(f"Random Forest MAE: {mae_rf:.2f} days")
    st.write(f"Linear Regression MAE: {mae_lr:.2f} days")
    st.write(f"SVR MAE: {mae_svr:.2f} days")
    
    # Actual vs Predicted plot
    fig, ax = plt.subplots()
    ax.scatter(models['y_test_duration'], y_pred_duration, alpha=0.3, label='Random Forest')
    ax.scatter(models['y_test_duration'], y_pred_duration_lr, alpha=0.3, label='Linear Regression')
    ax.scatter(models['y_test_duration'], y_pred_duration_svr, alpha=0.3, label='SVR')
    ax.plot([models['y_test_duration'].min(), models['y_test_duration'].max()], 
            [models['y_test_duration'].min(), models['y_test_duration'].max()], 'k--')
    ax.set_xlabel('Actual Duration (days)')
    ax.set_ylabel('Predicted Duration (days)')
    ax.set_title('Actual vs Predicted NCR Closure Duration')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Recurrence Prediction Performance")
    
    # Classification metrics
    y_pred_recurrence = models['classifier'].predict(models['X_test'])
    accuracy = accuracy_score(models['y_test_recurrence'], y_pred_recurrence)
    
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    st.text("Classification Report:")
    report = classification_report(models['y_test_recurrence'], y_pred_recurrence, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    
    # Confusion matrix
    cm = confusion_matrix(models['y_test_recurrence'], y_pred_recurrence)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
