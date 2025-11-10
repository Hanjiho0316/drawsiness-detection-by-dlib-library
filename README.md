# Eye Blink Detection using Dlib, MediaPipe, and Transfer Learning
![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Transfer_Learning-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A deep learning–based **eye blink and drowsiness detection system**  
that leverages **open datasets**, **Dlib**, and **MediaPipe** for robust eye tracking  
and uses **transfer learning** for accurate eye state classification.

---

## 🚀 Overview
This project improves upon traditional Dlib-based drowsiness detection by combining:
- **MediaPipe Face Mesh** for fast and accurate facial landmark extraction  
- **Dlib** as a fallback face detector in low-light or off-angle cases  
- **Transfer Learning** using a pre-trained CNN backbone (e.g., MobileNetV2 / EfficientNet)  
- **Open eye-blink datasets** (e.g., CEW, ZJU Eyeblink, RT-BENE) for robust model training

The trained model classifies **eye open/closed states** and detects **blink patterns**  
in real time through webcam input.

---

## ✨ Features
- 🔹 Multi-source facial landmark detection (MediaPipe + Dlib)
- 🔹 Eye-region cropping and normalization
- 🔹 Transfer-learned CNN model for eye-state classification
- 🔹 Real-time blink and drowsiness detection
- 🔹 Easily extendable to video fatigue analysis or driver monitoring systems

---

## 🧠 Methodology

### 1. Data
The model is trained using open datasets such as:
- **CEW (Closed Eyes in the Wild)**
- **ZJU Eyeblink Dataset**
- **RT-BENE (Real-Time Blink Estimation)**
These datasets provide eye images and blink annotations under various lighting and pose conditions.

### 2. Preprocessing
- MediaPipe Face Mesh → Extract 468 landmarks  
- Crop left/right eye patches using eye landmark coordinates  
- Resize and normalize each patch (e.g., 112×112)  
- Apply data augmentation for brightness, rotation, and occlusion variations

### 3. Model
- Backbone: Pre-trained CNN (MobileNetV2, ResNet18, or EfficientNet-B0)
- Fine-tuned on eye open/closed classification
- Loss: Binary Cross-Entropy (BCE)
- Optimizer: AdamW + LR scheduler
- Optional LSTM or Transformer for blink sequence modeling

### 4. Real-time Detection
- MediaPipe runs continuously on webcam frames
- Extracts eyes → feeds them to trained model
- Calculates open/closed probabilities
- Uses temporal smoothing to detect blink/drowsiness events
- Triggers on-screen or sound alerts when prolonged closure is detected

---

## 🧩 Installation
```bash
pip install opencv-python dlib mediapipe torch torchvision numpy
