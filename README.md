<div align="center">

# 🥔 Leafcheck

### Potato leaf disease detection with a Vision Transformer

Upload a photo of a potato leaf and get an instant diagnosis: **early blight**, **late blight** or **healthy**, with a confidence score for every class.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

</div>

---

## ✨ Overview

Leafcheck is an end-to-end deep learning project. A pretrained **Vision Transformer (ViT)** is fine-tuned on potato leaf images, then served through a **Flask** backend with a clean, responsive web interface.

Farmers and students can drag in a leaf photo, watch the scan run, and see how confident the model is about each possible condition.

<img width="1150" height="837" alt="image" src="https://github.com/user-attachments/assets/f8e25ecd-586d-492b-be2d-eb5958c8a5b9" />


## 🌿 What it detects

| Class | Meaning |
|---|---|
| **Early blight** | Fungal disease causing dark spots with concentric rings on older leaves |
| **Late blight** | Fast-spreading disease causing water-soaked, dark lesions on leaves |
| **Healthy** | No visible disease |

## 🧠 How it works

```
Leaf photo ──► Resize 224×224 + normalize ──► ViT-Base (fine-tuned) ──► Softmax ──► Diagnosis + confidence
```

| Item | Detail |
|---|---|
| Base model | `google/vit-base-patch16-224` |
| Task | 3-class image classification |
| Data split | 80% train, 20% test (seed 42) |
| Augmentation | Random horizontal flip (train only) |
| Loss / optimizer | Cross-entropy, AdamW (lr `2e-5`) |
| Training | 3 epochs, batch size 8 |
| Framework | PyTorch + Hugging Face Transformers |

The full training and evaluation pipeline is in the notebook `early_brite_week8_class1_transformer.ipynb`. It covers data loading, training curves, accuracy, the classification report, the confusion matrix and a prediction confidence histogram.

## 🖥️ Web interface
<img width="794" height="414" alt="Screenshot 2026-10-21 150126" src="https://github.com/user-attachments/assets/e06f397e-0bfa-45ff-bec1-36755e12b5de" />


- Drag and drop, or browse, to upload a leaf photo with instant preview
- Animated scan while the model analyzes the image
- Top diagnosis in large type, plus confidence bars for every class
- Input validation (file type and 10 MB size limit) with clear error messages
- Responsive layout, keyboard friendly, respects reduced-motion settings

## 📁 Project structure

```
earlybrite-model/
├── app.py                 # Flask backend (model loading + /predict API)
├── templates/
│   └── index.html         # Web interface
├── save_path/             # Fine-tuned model (not stored in the repo)
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── early_brite_week8_class1_transformer.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

**Windows (cmd)**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

The weights file is about 343 MB, which is above GitHub's file size limit, so it is not committed. Train it yourself with the notebook, or download it and place the three files in `save_path/`:

```
save_path/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

> 💡 Add a download link here, for example a Google Drive, Hugging Face Hub or GitHub Releases URL.

### 5. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## 🔌 API

### `POST /predict`

Send an image as `multipart/form-data` with the key `image`.

```bash
curl -X POST -F "image=@leaf.jpg" http://127.0.0.1:5000/predict
```

```json
{
  "prediction": "Potato___Early_blight",
  "confidence": 97.31,
  "probabilities": {
    "Potato___Early_blight": 97.31,
    "Potato___Late_blight": 2.12,
    "Potato___healthy": 0.57
  }
}
```

### `GET /health`

Returns the server status, the device in use (CPU or GPU) and the list of classes.

## 🛠️ Tech stack

- **Model:** PyTorch, Hugging Face Transformers, torchvision
- **Backend:** Flask, Pillow
- **Frontend:** HTML, CSS and vanilla JavaScript (no frameworks)
- **Training environment:** Google Colab

## 📊 Results

> Fill in after running the evaluation cells in the notebook.

| Metric | Score |
|---|---|
| Test accuracy | _add here_ |
| Precision / Recall / F1 | _add here_ |

## 🔮 Future improvements

- Support more crops and diseases
- Add Grad-CAM heatmaps to show which part of the leaf drove the prediction
- Mobile-friendly camera capture
- Deploy online (Hugging Face Spaces, Render or similar)
- Treatment suggestions for each detected disease

## ⚠️ Disclaimer

Leafcheck is an educational project. Results are statistical estimates from a single photo and can be wrong. For treatment or spraying decisions, consult a qualified agronomist or your local agricultural extension service.

## 🙌 Acknowledgements

- [Google ViT](https://huggingface.co/google/vit-base-patch16-224) for the pretrained Vision Transformer
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- The Early Brite program for the guided learning material

---

<div align="center">

Made with 🌱 and a lot of leaves

</div>
