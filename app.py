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
# Load Model and Class Mapping
# --------------------------------------------------
# Load trained model
model = load_model(MODEL_PATH)

# Load classes.json and map index → label
with open(CLASSES_PATH, "r") as f:
    class_map = json.load(f)

# Convert keys to int safely (your JSON is already { "0": "Label", ... })
idx_to_label = {int(k): v for k, v in class_map.items()}

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
    arr = preprocess_input(arr)  # ✅ Correct preprocessing for MobileNetV3
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
    uploaded_file = request.files.get('file')

    # Validate file upload
    if not uploaded_file or uploaded_file.filename == '':
        return "No image uploaded", 400

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)

    # Prepare image and predict
    img_array = prepare_image(file_path)
    preds = model.predict(img_array)
    predicted_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]) * 100)

    # Lookup label
    label = idx_to_label.get(predicted_idx, f"Unknown (index {predicted_idx})")

    # Debug print (optional)
    print(f"[INFO] Predicted: {label} | Confidence: {confidence:.2f}%")

    # Return result page
    return render_template(
        'result.html',
        image_path=file_path,
        prediction=label,
        confidence=round(confidence, 2)
    )


# --------------------------------------------------
# Run the Flask App
# --------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
