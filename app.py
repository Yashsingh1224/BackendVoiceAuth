import os
import pickle
import librosa
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model, scaler, and labels
MODEL_FILE = "voice_auth_model.pkl"
SCALER_FILE = "scaler.pkl"
LABELS_FILE = "labels.pkl"

model = pickle.load(open(MODEL_FILE, "rb"))
scaler = pickle.load(open(SCALER_FILE, "rb"))
labels = pickle.load(open(LABELS_FILE, "rb"))

app = Flask(__name__)

# Function to extract MFCC features from audio
def extract_features(file_path, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return np.ravel(mfccs)

@app.route("/authenticate", methods=["POST"])
def authenticate_user():
    if "file" not in request.files or "user" not in request.form:
        return jsonify({"error": "Missing file or user parameter"}), 400

    file = request.files["file"]
    user = request.form["user"]

    # Save the uploaded file temporarily
    file_path = "temp.wav"
    file.save(file_path)

    # Extract features and classify
    features = extract_features(file_path)
    features = scaler.transform([features])
    predicted_label = model.predict(features)[0]

    os.remove(file_path)  # Clean up

    # Check if the predicted label matches the requested user
    for label_name, label_id in labels.items():
        if label_id == predicted_label:
            if label_name == user:
                return jsonify({"authenticated": True, "user": user})
            else:
                return jsonify({"authenticated": False, "message": "Voice does not match user"})

    return jsonify({"authenticated": False, "message": "User not found"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
