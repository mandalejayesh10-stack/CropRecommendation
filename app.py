import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Constants for pickle paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Global variables for model resources
model = None
scaler = None
label_encoder = None

def load_resources():
    """Load the trained machine learning model, scaler, and label encoder."""
    global model, scaler, label_encoder
    
    # Check if files exist
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
        raise FileNotFoundError(
            "Required pickle files (model.pkl, scaler.pkl, label_encoder.pkl) not found. "
            "Please run 'train_model.py' first to train the model and save these files."
        )
        
    print("Loading ML model and preprocessors...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("Resources loaded successfully.")

# Load resources at startup
try:
    load_resources()
except Exception as e:
    print(f"Startup Warning: {e}")
    print("You will need to train the model using train_model.py before running web predictions.")

@app.route("/")
def home():
    """Render the homepage containing the soil and environment input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests from the input form."""
    global model, scaler, label_encoder
    
    # Reload resources if they were not loaded at startup (e.g. if trained after app start)
    if model is None or scaler is None or label_encoder is None:
        try:
            load_resources()
        except Exception as e:
            return f"Error: Model not trained yet. Details: {e}", 500

    try:
        # 1. Retrieve and parse inputs from form
        N = float(request.form.get("N", 0))
        P = float(request.form.get("P", 0))
        K = float(request.form.get("K", 0))
        temperature = float(request.form.get("temperature", 0))
        humidity = float(request.form.get("humidity", 0))
        ph = float(request.form.get("ph", 0))
        rainfall = float(request.form.get("rainfall", 0))
        
        inputs = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        # 2. Reshape features into a Pandas DataFrame with column names (suppresses scikit-learn feature name UserWarning)
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

        # 3. Apply standard scaling using the loaded scaler
        features_scaled = scaler.transform(features)

        # 4. Predict class using model
        prediction_encoded = model.predict(features_scaled)

        # 5. Decode predicted class to crop label
        recommended_crop = label_encoder.inverse_transform(prediction_encoded)[0]

        # 6. Render the result page
        return render_template("result.html", crop=recommended_crop, inputs=inputs)
        
    except ValueError as val_err:
        return f"Input parsing error. Please ensure all values are numeric: {val_err}", 400
    except Exception as e:
        return f"An error occurred during prediction: {e}", 500

if __name__ == "__main__":
    # Start the Flask app (reads PORT from env for cloud services, defaults to 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
