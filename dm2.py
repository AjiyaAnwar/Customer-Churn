import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import shap
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
        st.error("Model file 'churn_model.pkl' not found. Please ensure the model file is in the project directory.")
        st.stop()

# Load data for EDA
@st.cache_data
def load_eda_data():
    try:
        customer = pd.read_csv("Customer_Info.csv")
        service = pd.read_csv("Online_Services.csv").replace({'Yes': 1, 'No': 0})
        status = pd.read_csv("Status_Analysis.csv")
        # Create wide_customer by merging customer and status
        wide_customer = customer.merge(status[['customer_id', 'customer_status', 'churn_value', 'satisfaction_score', 'cltv']], on='customer_id', how='left')
        return customer, service, status, wide_customer
    except FileNotFoundError as e:
        st.error(f"Error: One or more data files not found in the 'data' directory. Please ensure 'Customer_Info.csv', 'Online_Services.csv', and 'Status_Analysis.csv' are present. ({str(e)})")
        st.stop()

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
            try:
                return pd.read_csv("Customer_Info.csv")
            except FileNotFoundError:
                st.error("Error: 'Customer_Info.csv' not found in the 'data' directory. Please ensure the file is present.")
                st.stop()
        
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

    # EDA for Uploaded CSV (only for Upload CSV mode)
    if input_mode == "Upload CSV":
        with st.expander("📊 Exploratory Data Analysis for Uploaded Data"):
            try:
                # Age and Gender Distribution
                if 'age' in input_df.columns and 'gender' in input_df.columns:
                    st.markdown("**Age and Gender Distribution**")
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    sns.histplot(data=input_df, x='age', hue='gender', multiple='stack',
                                 shrink=0.9, alpha=0.85, ax=axes[0], palette="viridis")
                    # Age group calculation
                    input_df['under_30'] = (input_df['age'] < 30).astype(int)
                    input_df['senior_citizen'] = (input_df['age'] >= 65).astype(int)
                    input_df['30-65'] = ((input_df['age'] >= 30) & (input_df['age'] < 65)).astype(int)
                    age_group = input_df[['under_30', 'senior_citizen', '30-65']].sum().reset_index(name='count')
                    age_group['index'] = ['Under 30', 'Senior Citizen', '30-65']
                    axes[1].pie(age_group['count'], labels=age_group['index'], autopct='%1.1f%%',
                                colors=["#ff9999", "#66b3ff", "#99ff99"], startangle=90,
                                wedgeprops={'edgecolor': 'black'})
                    st.pyplot(fig)
                else:
                    st.warning("Columns 'age' and/or 'gender' not found in uploaded CSV for age/gender distribution.")

                # Service Correlation Heatmap
                service_cols = ['phone_service_x', 'streaming_service', 'tech_service', 'unlimited_data', 'internet_type']
                available_service_cols = [col for col in service_cols if col in input_df.columns]
                if available_service_cols:
                    st.markdown("**Service Correlation Heatmap**")
                    service_matrix = input_df[available_service_cols].copy()
                    # Convert categorical to numeric
                    for col in service_matrix.columns:
                        if service_matrix[col].dtype == 'object':
                            service_matrix[col] = service_matrix[col].replace({'Yes': 1, 'No': 0})
                    service_matrix = pd.get_dummies(service_matrix, columns=['internet_type'] if 'internet_type' in service_matrix.columns else [], dtype=int)
                    corr_matrix = service_matrix.corr()
                    fig = plt.figure(figsize=(10, 6))
                    sns.heatmap(corr_matrix, cmap="mako", annot=True, fmt=".2f", linewidths=0.5,
                                vmin=-1, vmax=1, cbar=True, square=True, annot_kws={"size": 8})
                    st.pyplot(fig)
                else:
                    st.warning("No service-related columns found in uploaded CSV for correlation heatmap.")

                # Churn Categories & Satisfaction (if applicable)
                if 'churn_category' in input_df.columns and 'satisfaction_score' in input_df.columns:
                    st.markdown("**Churn Categories & Satisfaction**")
                    churn_data = input_df[input_df['Churn_Prediction'] == 1]  # Use predicted churn
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    churn_cat = churn_data.groupby('churn_category').size().reset_index(name='count')
                    ax[0].pie(churn_cat['count'], labels=churn_cat['churn_category'], autopct='%1.1f%%',
                              colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'],
                              startangle=140, wedgeprops={'edgecolor': 'black'})
                    sns.boxplot(data=churn_data, x='churn_category', y='satisfaction_score',
                                ax=ax[1], palette='coolwarm')
                    st.pyplot(fig)
                else:
                    st.warning("Columns 'churn_category' and/or 'satisfaction_score' not found in uploaded CSV for churn analysis.")
            except Exception as e:
                st.error(f"Error generating EDA: {str(e)}")

        # SHAP Feature Importance for Uploaded CSV
        with st.expander("📈 Feature Importance (SHAP Analysis)"):
            try:
                # Initialize SHAP explainer
                explainer = shap.Explainer(model, input_df_processed)
                shap_values = explainer(input_df_processed)
                
                # Plot SHAP summary (beeswarm plot)
                st.markdown("**SHAP Summary Plot (Feature Importance)**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, input_df_processed, show=False)
                st.pyplot(fig)
                
                # Plot SHAP bar plot for mean absolute SHAP values
                st.markdown("**SHAP Bar Plot (Mean Absolute Impact)**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, input_df_processed, plot_type="bar", show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating SHAP analysis: {str(e)}")

    # Model Performance on Test Set
    with st.expander("📈 Model Performance on Test Set"):
        y_pred = model.predict(x_test)
        st.markdown("**Confusion Matrix:**")
        confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
        st.dataframe(confusion_df.style.format("{:.2f}").set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'right')]}]
        ))
        
        st.markdown("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log the classification report in a structured format
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"Label: {label}")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.2f}")
            else:
                print(f"{label}: {metrics:.2f}")
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}").set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'right')]}]
        ))

# EDA / Insights Mode
elif app_mode == "EDA / Insights":
    st.header("🔍 Exploratory Data Analysis")
    customer, service, status, wide_customer = load_eda_data()
    model, expected_features, x_test, y_test = load_model()

    with st.expander("Customer Age and Gender Distribution"):
        try:
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
        except Exception as e:
            st.error(f"Error generating age/gender distribution: {str(e)}")

    with st.expander("Service Correlation Heatmap"):
        try:
            service_matrix = service.replace({'Yes': 1, 'No': 0})
            service_matrix = pd.get_dummies(service_matrix, columns=['internet_type'], dtype=int)
            corr_matrix = service_matrix.drop(['customer_id'], axis=1).corr()
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, cmap="mako", annot=True, fmt=".2f", linewidths=0.5,
                        vmin=-1, vmax=1, cbar=True, square=True, annot_kws={"size": 8})
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating service correlation heatmap: {str(e)}")

    with st.expander("Churn Categories & Satisfaction"):
        try:
            churn = status[status.customer_status == 'Churned'].drop(columns=['customer_status', 'churn_value'])
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
            churn_cat = churn.groupby('churn_category').size().reset_index(name='count')
            ax[0].pie(
                x=churn_cat['count'], labels=churn_cat['churn_category'], autopct='%1.1f%%',
                colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'}
            )
            ax[0].set_title("Churn Categories Distribution (%)", fontsize=11, fontweight="bold")
            category_order = ['Attitude', 'Competitor', 'Dissatisfaction', 'Other', 'Price']
            sns.boxplot(data=churn, x='churn_category', y='satisfaction_score', order=category_order, ax=ax[1], palette='coolwarm')
            ax[1].set_xticklabels(labels=category_order, fontsize=9, rotation=20)
            ax[1].set_title("Churn Category vs. Satisfaction Score", fontsize=11, fontweight="bold")
            ax[1].set_xlabel("")
            ax[1].set_ylabel("Satisfaction Score", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating churn categories plot: {str(e)}")

    with st.expander("Demographic Distribution"):
        try:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            gender_counts = customer['gender'].value_counts()
            sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax[0], palette='pastel')
            ax[0].set_title('Gender Distribution')
            ax[0].set_xlabel('Gender')
            ax[0].set_ylabel('Number of Customers')
            partner_counts = customer['partner'].value_counts()
            wedges, texts, autotexts = ax[1].pie(partner_counts, labels=partner_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90, wedgeprops={'edgecolor': 'black'})
            ax[1].set_title('Partner Status')
            ax[1].legend(wedges, partner_counts.index, title="Partner", loc="best")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating demographic plots: {str(e)}")

    with st.expander("Age Analysis"):
        try:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.boxplot(data=customer, x='gender', y='age', ax=ax[0], palette='Set2')
            ax[0].set_title('Age by Gender')
            ax[0].set_xlabel('Gender')
            ax[0].set_ylabel('Age')
            sns.histplot(customer['age'], kde=True, ax=ax[1], color='skyblue')
            ax[1].set_title('Age Distribution')
            ax[1].set_xlabel('Age')
            ax[1].set_ylabel('Number of Customers')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating age analysis plots: {str(e)}")

    with st.expander("Dependents & Marital Status"):
        try:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.barplot(data=customer, x='married', y='number_of_dependents', estimator=sum, ci=None, ax=ax[0], palette='muted')
            ax[0].set_title('Total Number of Dependents by Marital Status')
            ax[0].set_xlabel('Marital Status')
            ax[0].set_ylabel('Total Number of Dependents')
            partner_married = pd.crosstab(customer['partner'], customer['married'])
            partner_married.plot(kind='bar', stacked=True, ax=ax[1], color=['#a3c1ad', '#f7cac9'])
            ax[1].set_title('Partner vs. Married (Stacked)')
            ax[1].set_xlabel('Partner')
            ax[1].set_ylabel('Number of Customers')
            ax[1].legend(title='Married', loc='best')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating dependents/marital status plots: {str(e)}")

    with st.expander("Payment-Related Features Distribution"):
        try:
            payment_cols = [
                'monthly_charges', 'total_charges', 'total_revenue',
                'avg_monthly_long_distance_charges', 'total_long_distance_charges', 'total_extra_data_charges'
            ]
            available_cols = [col for col in payment_cols if col in status.columns]
            if not available_cols:
                available_cols = [col for col in payment_cols if col in customer.columns]
                payment_data = customer
            else:
                payment_data = status
            if available_cols:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                    titles = [
                        'Distribution of Monthly Wages', 'Distribution of Total Wages', 'Distribution of Total Income',
                        'Monthly Long Distance Fees', 'Total Long Distance Charges', 'Extra Data Charges'
                    ]
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
                    for ax, col, title, color in zip(axes.flat, available_cols, titles[:len(available_cols)], colors):
                        sns.histplot(payment_data[col], ax=ax, color=color, alpha=0.85, edgecolor='black')
                        ax.set_title(title, fontsize=11, fontweight='bold')
                        ax.set_xlabel("")
                        ax.set_ylabel("Frequency", fontsize=9)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                    for ax in axes.flat[len(available_cols):]:
                        ax.set_visible(False)
                    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
                    st.pyplot(fig)
            else:
                st.warning("No payment-related columns found in available datasets.")
        except Exception as e:
            st.error(f"Error generating payment features distribution: {str(e)}")

    with st.expander("Customer Status Analysis"):
        try:
            fig, ax = plt.subplots(1, 3, figsize=(14, 5))
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            status_group = status.groupby('customer_status').size().reset_index(name='count')
            ax[0].pie(
                status_group['count'], labels=status_group['customer_status'], autopct='%1.1f%%',
                colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'}
            )
            ax[0].set_title("Customer Status Distribution (%)", fontsize=11, fontweight="bold")
            label_order = ['Churned', 'Stayed', 'Joined']
            sns.boxplot(data=status, x='customer_status', y='satisfaction_score', order=label_order, ax=ax[1], palette='coolwarm')
            ax[1].set_title("Customer Status vs. Satisfaction Score", fontsize=11, fontweight="bold")
            ax[1].set_xlabel("")
            ax[1].set_ylabel("Satisfaction Score", fontsize=10)
            sns.boxplot(data=status, x='customer_status', y='cltv', order=label_order, ax=ax[2], palette='viridis')
            ax[2].set_title("Customer Status vs. CLTV", fontsize=11, fontweight="bold")
            ax[2].set_xlabel("")
            ax[2].set_ylabel("CLTV", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating customer status analysis: {str(e)}")

    with st.expander("Customer Status by Age and Gender"):
        try:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(
                data=wide_customer, x='age', hue='customer_status', multiple='stack', ax=ax[0],
                palette="viridis", edgecolor="black", alpha=0.9
            )
            ax[0].set_title('Customer Status and Age Distribution', fontsize=11, fontweight="bold")
            ax[0].set_xlabel("Age", fontsize=10)
            ax[0].set_ylabel("Number of Customers", fontsize=10)
            sns.countplot(
                data=wide_customer, x='gender', hue='customer_status', ax=ax[1], alpha=0.85,
                palette="Set2", edgecolor="black"
            )
            ax[1].set_title('Customer Status and Gender Distribution', fontsize=11, fontweight="bold")
            ax[1].set_xlabel("Gender", fontsize=10)
            ax[1].set_ylabel("Number of Customers", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating customer status by age/gender: {str(e)}")

    with st.expander("Feature Distributions (Histograms)"):
        try:
            num_cols = customer.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                fig, axes = plt.subplots(nrows=int(np.ceil(len(num_cols)/2)), ncols=2, figsize=(12, 4*int(np.ceil(len(num_cols)/2))))
                axes = axes.flatten() if len(num_cols) > 1 else [axes]
                for i, col in enumerate(num_cols):
                    sns.histplot(customer[col], kde=True, ax=axes[i], color='skyblue')
                    axes[i].set_title(f"Distribution of {col}")
                for j in range(i+1, len(axes)):
                    axes[j].set_visible(False)
                st.pyplot(fig)
            else:
                st.info("No numerical columns found in customer data for distribution plots.")
        except Exception as e:
            st.error(f"Error generating feature distributions: {str(e)}")

    with st.expander("Feature Importance (SHAP Analysis for Training Data)"):
        try:
            customer_data = pd.read_csv("Customer_Info.csv")
            service_data = pd.read_csv("Online_Services.csv").replace({'Yes': 1, 'No': 0})
            training_data = customer_data.merge(service_data, on='customer_id', how='left', suffixes=('', '_y'))
            columns_to_drop = [col for col in ['Churn', 'CLTV', 'CustomerID', 'customer_id'] if col in training_data.columns]
            training_data = training_data.drop(columns=columns_to_drop, errors='ignore')
            training_data_processed = pd.get_dummies(training_data, dtype=int)
            missing_cols = set(expected_features) - set(training_data_processed.columns)
            for col in missing_cols:
                training_data_processed[col] = 0
            extra_cols = set(training_data_processed.columns) - set(expected_features)
            training_data_processed = training_data_processed.drop(columns=extra_cols, errors='ignore')
            training_data_processed = training_data_processed[expected_features]
            
            explainer = shap.Explainer(model, training_data_processed)
            shap_values = explainer(training_data_processed)
            
            st.markdown("**SHAP Summary Plot (Feature Importance)**")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, training_data_processed, show=False)
            st.pyplot(fig)
            
            st.markdown("**SHAP Bar Plot (Mean Absolute Impact)**")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, training_data_processed, plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating SHAP analysis for training data: {str(e)}")

# About Mode
elif app_mode == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This application was built to help predict telecom customer churn using logistic regression.  
    It also provides powerful visual insights into customer behavior through EDA and feature importance analysis.
    
    **Features**:
    - New data prediction with tailored input fields
    - Manual or CSV input for batch predictions
    - Live predictions
    - EDA plots
    - Feature importance analysis using SHAP
    - Confusion matrix & classification report for test set
    """)
