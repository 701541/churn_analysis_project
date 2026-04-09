import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---- Page Config ----
st.set_page_config(page_title="Churn Prediction App", page_icon="📊", layout="wide")

# ---- Load Data ----
df = pd.read_csv('data/churn.csv')

# ---- Preprocess ----
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# ---- Train Model ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---- UI ----
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("📌 About")
st.sidebar.info("This app predicts customer churn using Machine Learning.")

st.markdown("### 🔍 Predict whether a customer is likely to churn or not")
num =st.slider("select number of customers,1 ,50 , 5 ")

if st.button("🔮 Predict Sample Customers"):
    predictions = model.predict(X_test[:num])

    result = pd.DataFrame({
        "Customer": [f"Customer {i+1}" for i in range(num)],
        "Prediction": ["❌ Churn" if p == 1 else "✅ Stay" for p in predictions]
    })

    st.success("Prediction Completed!")
    st.dataframe(result, use_container_width=True)

st.markdown("---")
st.markdown("<center>Made by Hiten 🚀</center>", unsafe_allow_html=True)