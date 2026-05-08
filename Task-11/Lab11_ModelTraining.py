# Lab 11 – Train/Test Split, Model, Prediction & Accuracy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and pre-process the dataset (same as Lab 10)
df = pd.read_csv('healthcare_dataset.csv')

# Convert numeric columns to int
int_cols = ['Patient_ID', 'Age', 'Blood_Pressure', 'Heart_Rate', 'Cholesterol_Level', 'BMI']
for col in int_cols:
    df[col] = df[col].astype(int)

# Convert date column
df['Follow_Up_Date'] = pd.to_datetime(df['Follow_Up_Date'])

# 2. Prepare features and target
# Drop ID, date, treatment plan (to avoid data leakage) and the target diagnosis
X = df.drop(columns=['Patient_ID', 'Follow_Up_Date', 'Treatment_Plan', 'Diagnosis'])
y = df['Diagnosis']

# Encode the categorical 'Gender' column
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])

# Encode the target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# 3. Train/Test Split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# 4. Choose and apply the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Testing / Predicting
y_pred = model.predict(X_test)

# 6. Display Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))