import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load dataset
df = pd.read_csv('data/churn.csv')

# Show first rows
print(df.head())

# Basic info
print(df.info())

# Missing values
print(df.isnull().sum())

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

# print("After cleaning:")
# print(df.info())
# sns.countplot(x='Churn', data=df)
# plt.show()
# sns.countplot(x='Contract', hue='Churn', data=df)
# plt.show()
# sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
# plt.show()
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data (IMPORTANT: add random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (FIXES WARNING + improves model)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

print(comparison)