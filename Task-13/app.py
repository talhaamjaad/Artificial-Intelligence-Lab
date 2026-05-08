import pickle, os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load pre-trained model and encoders if they exist, else train and save
if os.path.exists('diagnosis_model.pkl'):
    model = pickle.load(open('diagnosis_model.pkl', 'rb'))
    gender_enc = pickle.load(open('gender_encoder.pkl', 'rb'))
    target_enc = pickle.load(open('target_encoder.pkl', 'rb'))
else:
    df = pd.read_csv('healthcare_dataset.csv')
    int_cols = ['Patient_ID', 'Age', 'Blood_Pressure', 'Heart_Rate', 'Cholesterol_Level', 'BMI']
    for col in int_cols:
        df[col] = df[col].astype(int)
    X = df.drop(columns=['Patient_ID', 'Follow_Up_Date', 'Treatment_Plan', 'Diagnosis'])
    y = df['Diagnosis']
    gender_enc = LabelEncoder()
    X['Gender'] = gender_enc.fit_transform(X['Gender'])
    target_enc = LabelEncoder()
    y_enc = target_enc.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_enc)
    pickle.dump(model, open('diagnosis_model.pkl', 'wb'))
    pickle.dump(gender_enc, open('gender_encoder.pkl', 'wb'))
    pickle.dump(target_enc, open('target_encoder.pkl', 'wb'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        bp = int(request.form['bp'])
        hr = int(request.form['hr'])
        chol = int(request.form['chol'])
        bmi = float(request.form['bmi'])
        gender_encoded = gender_enc.transform([gender])[0]
        features = np.array([[age, gender_encoded, bp, hr, chol, bmi]])
        pred_encoded = model.predict(features)[0]
        diagnosis = target_enc.inverse_transform([pred_encoded])[0]
        return render_template('index.html', prediction=f"Predicted Diagnosis: {diagnosis}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)