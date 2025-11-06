import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# --------------------------------------------------
# Flask Configuration
# --------------------------------------------------
app = Flask(__name__)

MODEL_PATH = "mobilenetv3_after_finetuning.keras"
CLASSES_PATH = "classes.json"
UPLOAD_FOLDER = "static"

# --------------------------------------------------
# Load Model and Class Mapping (Once at startup)
# --------------------------------------------------
print("[INFO] Loading model and classes...")

try:
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

try:
    with open(CLASSES_PATH, "r") as f:
        class_map = json.load(f)
    idx_to_label = {int(k): v for k, v in class_map.items()}
    print("[INFO] Class mapping loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load class mapping: {e}")
    idx_to_label = {}

# Ensure static directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# Helper: Prepare Image
# --------------------------------------------------
def prepare_image(file_path, target_size=(224, 224)):
    """Load and preprocess image for MobileNetV3."""
    img = image.load_img(file_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route('/')
def index():
    """Render upload page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make predictions."""
    if model is None:
        return "Model not loaded. Check logs.", 500

    uploaded_file = request.files.get('file')
    if not uploaded_file or uploaded_file.filename == '':
        return "No image uploaded", 400

    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)

    try:
        img_array = prepare_image(file_path)
        preds = model.predict(img_array)
        predicted_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]) * 100)
        label = idx_to_label.get(predicted_idx, f"Unknown (index {predicted_idx})")

        print(f"[INFO] Predicted: {label} | Confidence: {confidence:.2f}%")

        return render_template(
            'result.html',
            image_path=file_path,
            prediction=label,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return f"Prediction failed: {e}", 500


# --------------------------------------------------
# Run the Flask App
# --------------------------------------------------
if __name__ == '__main__':
    # Render expects host=0.0.0.0 and a port (default 10000)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
