# Lab 10 – Data Pre-Processing
# Dealing with Null Values & Changing Datatypes to Int

import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('healthcare_dataset.csv')
print("First 5 rows:")
print(df.head())

print("\nData types BEFORE processing:")
print(df.dtypes)

# 2. Dealing with Null Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Since there are no nulls in this dataset, nothing is done.
# If there were nulls, we would use:
# df['Age'].fillna(df['Age'].median(), inplace=True)
# or df.dropna(inplace=True)

print("No null values found – no action needed.")

# 3. Changing Datatypes of Columns to "Int"
int_columns = ['Patient_ID', 'Age', 'Blood_Pressure', 'Heart_Rate',
               'Cholesterol_Level', 'BMI']

for col in int_columns:
    df[col] = df[col].astype(int)

# Optional: convert date column to datetime
df['Follow_Up_Date'] = pd.to_datetime(df['Follow_Up_Date'])

print("\nData types AFTER conversion:")
print(df.dtypes)

print("\nProcessed data (first 5 rows):")
print(df.head())

print("\nLab 10 pre-processing completed.")