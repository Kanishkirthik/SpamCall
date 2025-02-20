import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress CUDA/cuDNN warnings

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle  # For loading tokenizer

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load Pretrained Tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# ✅ Load Pretrained Model
modeln = load_model("model.h5")
maxlen = 189

@app.route("/")
def home():
    return "🚀 Keras ML Model API is running!"

@app.route("/predict", methods=["POST"])
def classify():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Text input is required"}), 400

        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding="post")

        prediction = modeln.predict(padded_sequence, verbose=0)[0][0]
        result = "fraud" if prediction > 0.5 else "normal"

        return jsonify({"prediction": result, "confidence": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)  # Disable debug mode for deployment
