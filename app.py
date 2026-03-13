from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model = None
scaler = None
label_encoders = None
feature_names = None

# Resolve model file from the project directory to avoid cwd-related failures.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, 'model', 'creditworthiness_model.pkl'),
    os.path.join(BASE_DIR, 'creditworthiness_model.pkl'),
]

model_path = next((p for p in MODEL_CANDIDATE_PATHS if os.path.exists(p)), None)

try:
    if model_path is None:
        raise FileNotFoundError(
            'creditworthiness_model.pkl not found in model/ or project root'
        )

    model_package = pickle.load(open(model_path, 'rb'))
   # model_package = pickle.load(model_path)
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoders = model_package['label_encoders']
    feature_names = model_package['feature_names']
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.form.to_dict()
        
        # Required fields
        required_fields = ['Age', 'Gender', 'Education', 'Income', 'Debt', 
                          'Credit_Score', 'Loan_Amount', 'Loan_Term', 
                          'Num_Credit_Cards', 'Payment_History', 
                          'Employment_Status', 'Residence_Type', 'Marital_Status']
        
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        # Create DataFrame
        input_data = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col])
                except:
                    return jsonify({'success': False, 'error': f'Invalid value for {col}'}), 400
        
        # Create engineered features
        input_data['Debt_to_Income'] = input_data['Debt'].astype(float) / (input_data['Income'].astype(float) + 1)
        input_data['Loan_to_Income'] = input_data['Loan_Amount'].astype(float) / (input_data['Income'].astype(float) + 1)
        input_data['Credit_Utilization'] = input_data['Debt'].astype(float) / (input_data['Credit_Score'].astype(float) + 1)
        
        # Ensure correct feature order
        X_input = input_data[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Determine status
        credit_status = "Creditworthy" if prediction == 1 else "Not Creditworthy"
        risk_level = "Low Risk" if probability > 0.7 else "Medium Risk" if probability > 0.4 else "High Risk"
        color_class = "success" if prediction == 1 else "danger"
        
        return jsonify({
            'success': True,
            'prediction': credit_status,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'color_class': color_class,
            'input_data': data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print(f"📍 Model loaded: {model is not None}")
    app.run(debug=True, host='0.0.0.0', port=5000)