# 🚑 EMSTRAP - Emergency Image Analysis System

EMSTRAP is an AI-powered Emergency Image Analysis System designed to automatically classify emergency situations from images. The system helps ambulance and emergency response platforms quickly identify the type of emergency and assist in prioritizing response.

---

## 📌 Features

- Emergency image classification using Deep Learning
- Fast prediction for real-time applications
- Easy integration with backend APIs
- PyTorch-based model
- Ready for deployment

---

## 📂 Emergency Classes

The model classifies images into the following categories:

- Accident
- Fire
- Human Emergency
- Non-Emergency

(The class mapping is stored in `models/classes.json`.)

---

## 🛠 Tech Stack

- Python 3.11+
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pillow

---

## 📁 Project Structure

```
EMSTRAP/
│
├── backend/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── raw/
│
├── models/
│   ├── emstrap_final.pth
│   ├── emstrap_best.pth
│   └── classes.json
│
├── reports/
│   ├── accuracy_curves.png
│   └── confusion_matrix.png
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── split_dataset.py
│   └── restore_raw.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Python Version

Python **3.11** (Recommended)

---

# Installation

Clone the repository

```bash
git clone https://github.com/Janhavi-S803/EMSTRAP.git
cd EMSTRAP
```

Create a virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Model Files

```
models/
│
├── emstrap_final.pth
├── emstrap_best.pth
└── classes.json
```

---

# Loading the Model

Example:

```python
import torch

model = torch.load("models/emstrap_final.pth", map_location="cpu")
model.eval()
```

*(If your project uses `load_state_dict`, update this example to match your implementation.)*

---

# Input Requirements

- Image Format: JPG / JPEG / PNG
- RGB Image
- Image Size: **224 × 224 pixels**
- Normalization: Same preprocessing used during training

---

# Output

The model predicts one of the emergency classes.

Example:

```
Input Image

↓

Prediction

Fire

Confidence: 98.42%
```

---

# Example API Flow

```
Client

↓

Backend API

↓

EMSTRAP Model

↓

Predicted Emergency Class

↓

Response to Client
```

---

# Reports

Training reports are available inside

```
reports/
```

including

- Accuracy Curves
- Confusion Matrix

---

# Deployment Notes

For deployment only the following files are required:

```
models/
scripts/
requirements.txt
classes.json
```

The dataset is **not required** for inference.

---

# Future Improvements

- Video stream support
- Live CCTV integration
- Multi-label emergency detection
- Emergency severity estimation
- GPS-based emergency routing
- Edge device deployment

---

# Author

**Janhavi S**

B.Tech Artificial Intelligence & Machine Learning

GitHub:
https://github.com/Janhavi-S803

LinkedIn:
https://www.linkedin.com/in/janhavi-s-54a563335

---

# License

This project is developed for educational and research purposes.
