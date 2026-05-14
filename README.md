# Pneumonia Detection with VGG16 Transfer Learning

A deep learning web application that detects pneumonia from chest X-ray images using VGG16 transfer learning, deployed with a Flask API.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 86.06% |
| Test Loss | 0.3332 |
| ROC-AUC Score | 0.9265 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.82 | 0.81 | 0.81 | 234 |
| Pneumonia | 0.89 | 0.89 | 0.89 | 390 |
| **Accuracy** | | | **0.86** | **624** |

## Model Architecture

- **Backbone:** VGG16 (ImageNet pretrained, frozen)
- **Head:** Flatten → Dense(1024, ReLU) → Dropout(0.6) → Dense(2, Softmax)
- **Optimizer:** Adam (lr=5e-5)
- **Loss:** Categorical Crossentropy
- **Total Parameters:** 40,407,874

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle

| Split | Images |
|-------|--------|
| Train | 5,216 |
| Validation | 16 |
| Test | 624 |

## Project Structure

├── Traning.ipynb        # Model training notebook
├── flaskii.py           # Flask API backend
├── index.html           # Frontend UI
├── pneumonia_vgg16.keras  # Saved model
└── README.md



## Setup & Run

```bash
pip install tensorflow flask numpy pillow
python flaskii.py
Then open http://localhost:5000 in your browser.

Usage
Upload a chest X-ray image
The model returns: Normal or Pneumonia
Tech Stack
Python, TensorFlow / Keras
VGG16 Transfer Learning
Flask, HTML/CSS
scikit-learn, NumPy, Matplotlib

