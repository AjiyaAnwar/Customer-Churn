import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Customer Churn App", layout="wide")
st.title("📊 Customer Churn & CLTV Prediction App")

# Sidebar Navigation
st.sidebar.header("🔍 Navigation")
app_mode = st.sidebar.radio("Choose App Mode:", ["New Data Prediction", "Prediction", "EDA / Insights", "About"])

# Load model and feature names
@st.cache_resource
def load_model():
    model_path = "churn_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found. Please run the training script first.")
        st.stop()

# Load data for EDA
@st.cache_data
def load_eda_data():
    customer = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Customer_Info.csv")
    service = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Online_Services.csv").replace({'Yes': 1, 'No': 0})
    status = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Status_Analysis.csv")
    return customer, service, status

# New Data Prediction Mode
if app_mode == "New Data Prediction":
    st.header("🔮 New Customer Data Prediction")
    
    # Load model
    model, expected_features, x_test, y_test = load_model()

    # Input form for new data
    st.sidebar.subheader("📋 Enter Customer Details")
    
    # Define input fields based on model features
    user_data = {}
    user_data['age'] = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    user_data['number_of_dependents'] = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    user_data['phone_service_x'] = st.sidebar.selectbox("Phone Service", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_data['streaming_service'] = st.sidebar.number_input("Streaming Services (TV + Movies + Music)", min_value=0, max_value=3, value=0, step=1)
    user_data['tech_service'] = st.sidebar.number_input("Tech Services (Security + Backup + Protection + Support)", min_value=0, max_value=4, value=0, step=1)
    user_data['unlimited_data'] = st.sidebar.selectbox("Unlimited Data", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_data['number_of_referrals'] = st.sidebar.number_input("Number of Referrals", min_value=0, max_value=50, value=0, step=1)
    user_data['satisfaction_score'] = st.sidebar.number_input("Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    user_data['contract'] = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    user_data['internet_type'] = st.sidebar.selectbox("Internet Type", ["Fiber optic", "DSL", "Cable", "None"])

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Preprocess input
    input_df_processed = pd.get_dummies(input_df, columns=['contract', 'internet_type'], dtype=int)
    
    # Align columns with training data
    missing_cols = set(expected_features) - set(input_df_processed.columns)
    for col in missing_cols:
        input_df_processed[col] = 0
    extra_cols = set(input_df_processed.columns) - set(expected_features)
    input_df_processed = input_df_processed.drop(columns=extra_cols, errors='ignore')
    input_df_processed = input_df_processed[expected_features]

    # Predict
    if st.sidebar.button("Predict"):
        predictions = model.predict(input_df_processed)
        prediction_proba = model.predict_proba(input_df_processed)
        input_df['Churn_Prediction'] = predictions
        input_df['Churn_Probability'] = prediction_proba[:, 1]

        st.subheader("Prediction Results")
        # Apply highlight_max only to numerical columns
        numerical_cols = input_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        styled_df = input_df.style.highlight_max(subset=numerical_cols, axis=0, color="lightgreen")
        st.dataframe(styled_df)
        st.success(f"Churn Prediction: {'Yes' if predictions[0] == 1 else 'No'} (Probability: {prediction_proba[0, 1]:.2%})")

# Prediction Mode
elif app_mode == "Prediction":
    input_mode = st.sidebar.radio("Select Input Mode:", ["Manual Entry", "Upload CSV"])
    
    if input_mode == "Manual Entry":
        @st.cache_data
        def load_data():
            return pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Customer_Info.csv")
        
        data = load_data()
        columns_to_drop = [col for col in ['Churn', 'CLTV', 'CustomerID'] if col in data.columns]
        feature_cols = data.drop(columns=columns_to_drop).columns
        user_data = {}
        for col in feature_cols:
            if data[col].dtype == 'object':
                user_data[col] = st.sidebar.selectbox(f"{col}", sorted(data[col].dropna().unique()))
            else:
                if col in ['number_of_dependents', 'age']:
                    user_data[col] = st.sidebar.number_input(
                        f"{col}", min_value=int(data[col].min()), max_value=int(data[col].max()),
                        value=int(data[col].mean()), step=1)
                else:
                    user_data[col] = st.sidebar.number_input(
                        f"{col}", min_value=float(data[col].min()), max_value=float(data[col].max()),
                        value=float(data[col].mean()))
        input_df = pd.DataFrame([user_data])
    else:
        uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")
        else:
            st.warning("Please upload a CSV file.")
            st.stop()

    model, expected_features, x_test, y_test = load_model()
    
    st.subheader("🔮 Prediction Results")
    input_df_processed = pd.get_dummies(input_df, dtype=int)
    missing_cols = set(expected_features) - set(input_df_processed.columns)
    for col in missing_cols:
        input_df_processed[col] = 0
    extra_cols = set(input_df_processed.columns) - set(expected_features)
    input_df_processed = input_df_processed.drop(columns=extra_cols, errors='ignore')
    input_df_processed = input_df_processed[expected_features]
    
    predictions = model.predict(input_df_processed)
    prediction_proba = model.predict_proba(input_df_processed)
    input_df['Churn_Prediction'] = predictions
    input_df['Churn_Probability'] = prediction_proba[:, 1]
    # Apply highlight_max only to numerical columns
    numerical_cols = input_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    styled_df = input_df.style.highlight_max(subset=numerical_cols, axis=0, color="lightgreen")
    st.dataframe(styled_df)
    st.success("Prediction complete.")

    with st.expander("📈 Model Performance on Test Set"):
        y_pred = model.predict(x_test)
        st.markdown("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
        st.markdown("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

# EDA / Insights Mode
elif app_mode == "EDA / Insights":
    st.header("🔍 Exploratory Data Analysis")
    customer, service, status = load_eda_data()

    with st.expander("Customer Age and Gender Distribution"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(data=customer, x='age', hue='gender', multiple='stack',
                     shrink=0.9, alpha=0.85, ax=axes[0], palette="viridis")
        age_group = customer[['under_30', 'senior_citizen']].replace({'Yes': 1, 'No': 0})
        age_group['30-65'] = 1 - (age_group.under_30 + age_group.senior_citizen)
        age_group = age_group.sum().reset_index(name='count')
        axes[1].pie(age_group['count'], labels=age_group['index'], autopct='%1.1f%%',
                    colors=["#ff9999", "#66b3ff", "#99ff99"], startangle=90,
                    wedgeprops={'edgecolor': 'black'})
        st.pyplot(fig)

    with st.expander("Service Correlation Heatmap"):
        service_matrix = service.replace({'Yes': 1, 'No': 0})
        service_matrix = pd.get_dummies(service_matrix, columns=['internet_type'], dtype=int)
        corr_matrix = service_matrix.drop(['customer_id'], axis=1).corr()
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, cmap="mako", annot=True, fmt=".2f", linewidths=0.5,
                    vmin=-1, vmax=1, cbar=True, square=True, annot_kws={"size": 8})
        st.pyplot(fig)

    with st.expander("Churn Categories & Satisfaction"):
        churn = status[status.customer_status == 'Churned'].drop(columns=['customer_status', 'churn_value'])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        churn_cat = churn.groupby('churn_category').size().reset_index(name='count')
        ax[0].pie(churn_cat['count'], labels=churn_cat['churn_category'], autopct='%1.1f%%',
                  colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'],
                  startangle=140, wedgeprops={'edgecolor': 'black'})
        sns.boxplot(data=churn, x='churn_category', y='satisfaction_score',
                    ax=ax[1], palette='coolwarm')
        st.pyplot(fig)

# About Mode
elif app_mode == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This application was built to help predict telecom customer churn using logistic regression.  
    It also provides powerful visual insights into customer behavior through EDA.
    
    **Features**:
    - New data prediction with tailored input fields
    - Manual or CSV input for batch predictions
    - Live predictions
    - EDA plots
    - Confusion matrix & classification report
    """)