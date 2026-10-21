import io
import os
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "save_path")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
id2label = model.config.id2label
print("Model loaded on", device, "| Classes:", id2label)


def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def find_index_html():
    # Looks in templates/ first, then next to app.py
    for path in (os.path.join(BASE_DIR, "templates", "index.html"),
                 os.path.join(BASE_DIR, "index.html")):
        if os.path.exists(path):
            return path
    return None


def predict_image(image):
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        probs = F.softmax(model(pixel_values=pixel_values).logits, dim=1)[0]
    top = int(torch.argmax(probs))
    return {
        "prediction": id2label[top],
        "confidence": round(float(probs[top]) * 100, 2),
        "probabilities": {id2label[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)},
    }


@app.route("/")
def index():
    path = find_index_html()
    if path is None:
        return "index.html not found. Put it in the templates folder or next to app.py.", 404
    return send_file(path)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device), "classes": list(id2label.values())})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send a file with key 'image'."}), 400
    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400
    try:
        return jsonify(predict_image(Image.open(io.BytesIO(file.read()))))
    except UnidentifiedImageError:
        return jsonify({"error": "Could not read the image file."}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large. Max size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)