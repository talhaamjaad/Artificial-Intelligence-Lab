# app.py - Lab 12: Flask Application for Diagnosis Prediction
import pickle
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ----------------------------------------------
# Load or train the model
# ----------------------------------------------
MODEL_PATH = "diagnosis_model.pkl"
GENDER_ENCODER_PATH = "gender_encoder.pkl"
TARGET_ENCODER_PATH = "target_encoder.pkl"

# If model files exist, load them; otherwise train and save
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
    gender_encoder = pickle.load(open(GENDER_ENCODER_PATH, 'rb'))
    target_encoder = pickle.load(open(TARGET_ENCODER_PATH, 'rb'))
else:
    # Load and preprocess the dataset (same as Lab 10 & 11)
    df = pd.read_csv('healthcare_dataset.csv')

    # Convert numeric columns to int
    int_cols = ['Patient_ID', 'Age', 'Blood_Pressure', 'Heart_Rate', 'Cholesterol_Level', 'BMI']
    for col in int_cols:
        df[col] = df[col].astype(int)

    # Prepare features and target
    X = df.drop(columns=['Patient_ID', 'Follow_Up_Date', 'Treatment_Plan', 'Diagnosis'])
    y = df['Diagnosis']

    # Encode Gender
    gender_encoder = LabelEncoder()
    X['Gender'] = gender_encoder.fit_transform(X['Gender'])

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    # Save model and encoders
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    pickle.dump(gender_encoder, open(GENDER_ENCODER_PATH, 'wb'))
    pickle.dump(target_encoder, open(TARGET_ENCODER_PATH, 'wb'))

print("Model ready for predictions.")
print("Available target classes:", list(target_encoder.classes_))

# ----------------------------------------------
# Routes
# ----------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        age = int(request.form['age'])
        gender = request.form['gender']            # "Male", "Female", "Other"
        blood_pressure = int(request.form['bp'])
        heart_rate = int(request.form['hr'])
        cholesterol = int(request.form['chol'])
        bmi = float(request.form['bmi'])

        # Transform gender
        gender_encoded = gender_encoder.transform([gender])[0]

        # Create feature array (order must match training: Age, Gender, Blood_Pressure, Heart_Rate, Cholesterol_Level, BMI)
        features = np.array([[age, gender_encoded, blood_pressure, heart_rate, cholesterol, bmi]])

        # Predict
        pred_encoded = model.predict(features)[0]
        diagnosis = target_encoder.inverse_transform([pred_encoded])[0]

        return render_template('index.html', prediction=f"Predicted Diagnosis: {diagnosis}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)