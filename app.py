from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the model and scaler
try:
    rf_model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = scaler.feature_names_in_
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model or scaler file not found. Please upload 'model.pkl' and 'scaler.pkl'.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data, makes a prediction, and returns the result.
    """
    try:
        data = request.json
        
        # Convert input JSON to a DataFrame
        df_input = pd.DataFrame([data], columns=data.keys())

        # Preprocess the data to match the training format
        df_preprocessed = df_input.copy()
        df_preprocessed = df_preprocessed.reindex(columns=feature_names, fill_value=0)
        df_scaled = scaler.transform(df_preprocessed)
        
        # Make prediction and get probability
        prediction = rf_model.predict(df_scaled)[0]
        prediction_proba = rf_model.predict_proba(df_scaled)
        
        # Format the response
        result_string = "Heart Disease" if prediction == 1 else "No Heart Disease"
        risk_score = f"{prediction_proba[0][1]*100:.2f}%"

        response = {
            'prediction': result_string,
            'score': risk_score,
            'top_factors': [] # Simplified, LIME is not included in this server-side script
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask server on port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
