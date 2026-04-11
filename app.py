import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Churn Prediction Pro", layout="wide")


st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Churn Prediction App")
st.markdown("Upload your dataset → Map columns → Get churn insights")

st.sidebar.header("⚙️ Settings")
show_data = st.sidebar.checkbox("Show Raw Data", True)
show_charts = st.sidebar.checkbox("Show Charts", True)
show_risk = st.sidebar.checkbox("Show High Risk Customers", True)

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")


    if show_data:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

    
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


    new_df = pd.DataFrame()
    new_df["Age"] = df[age_col]
    new_df["Tenure"] = df[tenure_col]
    new_df["MonthlyCharges"] = df[charge_col]
    new_df["Churn"] = df[churn_col]


    new_df = new_df.dropna()

    
    drop_cols = ["Name", "CustomerID"]
    for col in drop_cols:
        if col in new_df.columns:
            new_df = new_df.drop(col, axis=1)


    new_df["Churn"] = new_df["Churn"].astype('category').cat.codes

    
    X = pd.get_dummies(new_df.drop("Churn", axis=1), drop_first=True)
    y = new_df["Churn"]

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    new_df["Prediction"] = predictions
    new_df["Churn Probability"] = probabilities


    st.subheader("📊 Dashboard")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", len(new_df))
    m2.metric("Churn Customers", int(np.sum(predictions)))
    m3.metric("Churn Rate (%)", round(np.mean(predictions)*100, 2))


    st.subheader("📋 Results")
    st.dataframe(new_df.head(100))

    
    csv = new_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results", csv, "churn_results.csv")

    if show_charts:
        st.subheader("📈 Visual Insights")

        c1, c2 = st.columns(2)

        with c1:
            st.write("Churn Distribution")
            st.bar_chart(new_df["Prediction"].value_counts())

        with c2:
            st.write("Churn Probability Trend")
            st.line_chart(new_df["Churn Probability"])

  
    if show_risk:
        st.subheader("⚠️ High Risk Customers (Prob > 0.7)")

        high_risk = new_df[new_df["Churn Probability"] > 0.7]

        if len(high_risk) > 0:
            st.dataframe(high_risk)
        else:
            st.success("No high-risk customers 🎉")

else:
    st.info("👆 Upload a CSV file to start")
