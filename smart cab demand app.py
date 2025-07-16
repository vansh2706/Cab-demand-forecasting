import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Assuming preprocess and utils are in the same directory or accessible via PYTHONPATH
# For a simple setup, you might import directly:
from preprocess import preprocess_for_prediction
from utils import load_model

app = Flask(_name_)

# Load the model once when the app starts
MODEL_PATH = os.path.join('models', 'xgb_model.pkl')
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None if loading fails


@app.route('/')
def home():
    """
    Renders the optional UI page.
    """
    return render_template('index.html')


@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    """
    API endpoint to predict cab demand.
    Expects JSON input with relevant features.
    Example Input:
    {
        "timestamp": "2025-07-16 18:00:00",
        "latitude": 26.54,
        "longitude": 77.03
        # Add other features if your model uses them, e.g., 'weather_condition'
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    data = request.get_json(force=True)

    # Convert raw input to a DataFrame suitable for preprocessing
    # Ensure all expected raw features are present, even if dummy
    try:
        input_df = pd.DataFrame([data])

        # Preprocess the input data for prediction
        # This function should ensure the input features match the training features
        processed_features = preprocess_for_prediction(input_df)

        # Make prediction
        prediction = model.predict(processed_features)

        return jsonify({'predicted_demand': float(prediction[0])})
    except KeyError as e:
        return jsonify({"error": f"Missing expected input feature: {e}. Please ensure timestamp, latitude, and longitude are provided."}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if _name_ == '_main_':
    # For development: run directly
    # For production: use a WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)
