from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import requests

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

# Load the model and scaler
try:
    rf_model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = list(scaler.feature_names_in_)  # Ensure it's a list
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model or scaler file not found. Please upload 'model.pkl' and 'scaler.pkl'.")
    exit()

# Endpoint to get current ngrok URL
@app.route('/ngrok-url', methods=['GET'])
def get_ngrok_url():
    try:
        tunnels = requests.get('http://127.0.0.1:4040/api/tunnels').json()
        public_url = tunnels['tunnels'][0]['public_url']
        return jsonify({"ngrok_url": public_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data, makes a prediction, and returns the result.
    """
    try:
        data = request.json
        
        # Convert input JSON into DataFrame with correct feature order
        df_input = pd.DataFrame([data])
        df_input = df_input.reindex(columns=feature_names, fill_value=0)

        # Keep DataFrame format when scaling
        df_scaled = pd.DataFrame(
            scaler.transform(df_input),
            columns=feature_names
        )

        # Make prediction and get probability
        prediction = rf_model.predict(df_scaled)[0]
        prediction_proba = rf_model.predict_proba(df_scaled)

        # Format the response
        result_string = "Heart Disease" if prediction == 1 else "No Heart Disease"
        risk_score = f"{prediction_proba[0][1]*100:.2f}%"

        response = {
            'prediction': result_string,
            'score': risk_score,
            'top_factors': []  # Simplified, no LIME here
        }
        print("Received data:", data)
        print("Prediction:", prediction, "Score:", risk_score)
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)