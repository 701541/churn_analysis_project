import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 AI-Based Customer Churn Prediction System")
st.markdown("Upload your dataset → Map columns → Get churn insights + retention strategy")

# Sidebar
st.sidebar.header("⚙️ Settings")
show_data = st.sidebar.checkbox("Show Raw Data", True)
show_charts = st.sidebar.checkbox("Show Charts", True)
show_risk = st.sidebar.checkbox("Show High Risk Customers", True)

# Upload file
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show raw data
    if show_data:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

    # Column Mapping
    st.subheader("🔄 Column Mapping")

    columns = df.columns.tolist()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age_col = st.selectbox("Age Column", columns)

    with col2:
        tenure_col = st.selectbox("Tenure Column", columns)

    with col3:
        charge_col = st.selectbox("Monthly Charges", columns)

    with col4:
        churn_col = st.selectbox("Churn Column", columns)

    # Create new dataframe
    new_df = pd.DataFrame()
    new_df["Age"] = df[age_col]
    new_df["Tenure"] = df[tenure_col]
    new_df["MonthlyCharges"] = df[charge_col]
    new_df["Churn"] = df[churn_col]

    new_df = new_df.dropna()

    # Encode target
    new_df["Churn"] = new_df["Churn"].astype('category').cat.codes

    # Features & target
    X = pd.get_dummies(new_df.drop("Churn", axis=1), drop_first=True)
    y = new_df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Accuracy
    acc = accuracy_score(y_test, predictions)
    st.success(f"✅ Model Accuracy: {round(acc*100, 2)}%")

    # Risk levels
    def risk_level(p):
        if p > 0.7:
            return "High Risk"
        elif p > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    result_df = X_test.copy()
    result_df["Actual"] = y_test.values
    result_df["Prediction"] = predictions
    result_df["Churn Probability"] = probabilities
    result_df["Risk Level"] = result_df["Churn Probability"].apply(risk_level)

    # Retention Suggestions
    def suggest(row):
        if row["Risk Level"] == "High Risk":
            return "Give discount + personal call"
        elif row["Risk Level"] == "Medium Risk":
            return "Offer upgrade or engagement plan"
        else:
            return "No action needed"

    result_df["Suggestion"] = result_df.apply(suggest, axis=1)

    # Dashboard
    st.subheader("📊 Dashboard")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", len(result_df))
    m2.metric("High Risk Customers", (result_df["Risk Level"] == "High Risk").sum())
    m3.metric("Churn Rate (%)", round(result_df["Prediction"].mean()*100, 2))

    # Results
    st.subheader("📋 Prediction Results")
    st.dataframe(result_df.head(100))

    # Download
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results", csv, "churn_results.csv")

    # Charts
    if show_charts:
        st.subheader("📈 Visual Insights")

        c1, c2 = st.columns(2)

        with c1:
            st.write("Risk Distribution")
            st.bar_chart(result_df["Risk Level"].value_counts())

        with c2:
            st.write("Churn Probability Trend")
            st.line_chart(result_df["Churn Probability"])

    # Feature Importance
    st.subheader("🔥 Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(importance.sort_values(ascending=False))

    # High Risk Customers
    if show_risk:
        st.subheader("⚠️ High Risk Customers (Prob > 0.7)")

        high_risk = result_df[result_df["Churn Probability"] > 0.7]

        if len(high_risk) > 0:
            st.dataframe(high_risk)
        else:
            st.success("No high-risk customers 🎉")

else:
    st.info("👆 Upload a CSV file to start")