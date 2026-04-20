from flask import Flask, request, jsonify
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load TFLite model with diagnostic logging
try:
    print("⏳ Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path="garbage_classifier.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODEL: {str(e)}")
    # We allow the error to bubble up so Gunicorn logs it clearly
    raise e

# Match your training labels
classes = ["Biodegradable", "Non-biodegradable", "Recyclable"]

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Adjusted to match EfficientNetB0 training size
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Flutter sends the image under the key 'file'
        img_file = request.files.get("file") or request.files.get("image")
        
        if not img_file:
            return jsonify({"error": "No image uploaded. Make sure to send as 'file' or 'image'."}), 400

        image = Image.open(img_file)

        processed = preprocess(image).astype(np.float32)
        
        # 1. Set Input Tensor
        interpreter.set_tensor(input_details[0]['index'], processed)
        
        # 2. Run Inference
        interpreter.invoke()
        
        # 3. Get Output Tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        # Flutter app expects 'label' and 'confidence'
        return jsonify({
            "label": classes[class_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ IMPORTANT FIX FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
