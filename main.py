# ===============================
# Project 1: Customer Retention & Churn Analysis
# ===============================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# 2️⃣ Load Dataset
# ===============================
# Make sure your CSV path is correct
df = pd.read_csv(r"C:\Users\vishw\OneDrive\Desktop\python\customer retention and churn analysis\OnlineRetail.csv",
                 encoding='ISO-8859-1')
# Take a sample of the large CSV
df_sample = df.sample(n=5000, random_state=42)  # 5000 rows
df_sample.to_csv("OnlineRetail_sample.csv", index=False)


print("Dataset Loaded Successfully!")
print(df.head())

# ===============================
# 3️⃣ Data Cleaning
# ===============================
# Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove negative quantities
df = df[df['Quantity'] > 0]

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("Data cleaned successfully!")

# ===============================
# 4️⃣ RFM Calculation
# ===============================
# Reference date = max InvoiceDate + 1 day
ref_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by CustomerID
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',                                # Frequency
    'TotalPrice': 'sum'                                  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("RFM calculation done!")
print(rfm.head())

# ===============================
# 5️⃣ Define Churn
# ===============================
# Customers with Recency > 90 days considered churned
rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
print(rfm['Churn'].value_counts())

# ===============================
# 6️⃣ Exploratory Analysis
# ===============================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(rfm['Recency'], bins=30, kde=True, color='blue')
plt.title("Customer Recency Distribution")

plt.subplot(1,2,2)
sns.histplot(rfm['Monetary'], bins=30, kde=True, color='green')
plt.title("Customer Monetary Distribution")

plt.tight_layout()
plt.show()

# Churn rate
churn_rate = rfm['Churn'].mean() * 100
print(f"Churn Rate: {churn_rate:.2f}%")

# ===============================
# 7️⃣ Machine Learning Model (Optional)
# ===============================
X = rfm[['Recency','Frequency','Monetary']]
y = rfm['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 8️⃣ Feature Importance
# ===============================
importances = model.feature_importances_
features = ['Recency','Frequency','Monetary']

plt.figure(figsize=(6,4))
plt.bar(features, importances, color='orange')
plt.title("Feature Importance for Churn Prediction")
plt.show()

# ===============================
# 9️⃣ Save Processed Data
# ===============================
rfm.to_csv("customer_churn_analysis.csv", index=False)
print("Final RFM dataset with churn label saved as customer_churn_analysis.csv")
