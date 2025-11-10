# MultiModal Eye Blink, Gaze, and Head Pose Detection
![Python](https://img.shields.io/badge/python-3.x-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-Transfer_Learning-orange.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green.svg)  
![Dlib](https://img.shields.io/badge/Dlib-Facial_Landmarks-blue.svg)  
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face_Mesh-purple.svg)  
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A **real-time multimodal system** for **eye blink detection**, **pupil-based gaze estimation**, and **head pose tracking**.  
It leverages **Dlib 68-point landmarks**, **MediaPipe 468-point landmarks**, and **ResNet18-based CNN features** for robust multimodal inference.

---

## 🚀 Overview

This project extends traditional blink detection by combining:

- **MediaPipe Face Mesh** (468 points) for high-speed facial landmark extraction  
- **Dlib 68-point landmarks** for robustness under occlusion or poor lighting  
- **Eye-region cropping** for CNN-based feature extraction  
- **Multimodal fusion**: eye-crop CNN features + full-face landmark features  
- **Pupil position tracking**: normalized relative x-position (-1 left → 1 right)  
- **Head pose estimation**: Yaw (left/right) and Pitch (up/down) using solvePnP  

The system runs in **real-time** on webcam input and provides both **visual feedback** and **numeric outputs**.

---

## ✨ Features

- 🔹 Multimodal input: eye-crop images + full-face landmarks  
- 🔹 Transfer learning with **ResNet18** backbone  
- 🔹 Real-time blink detection (Open/Closed)  
- 🔹 Pupil-based gaze estimation (-1.0 to 1.0)  
- 🔹 Head pose angles (Yaw, Pitch)  
- 🔹 Annotated webcam display with blink status, count, gaze, and head angles  
- 🔹 GPU/CPU compatible  

---

## 🧠 Methodology

### 1. Dataset
The model is primarily trained on the **CEW Dataset**:

| Dataset | Source | Description |
|---------|--------|-------------|
| CEW | [Kaggle](https://www.kaggle.com/datasets/ahamedfarouk/cew-dataset) | Labeled open/closed eye images for training |
| Optional Validation | [Drowsiness Detection Dataset](https://www.kaggle.com/datasets/norannabil/drowness-detection) | Evaluate blink-based drowsiness |

- Open eyes labeled as `1`, closed eyes as `0`  
- Weighted CrossEntropyLoss used to handle class imbalance  

---

### 2. Preprocessing

1. **Facial landmarks extraction**:  
   - Dlib: 68 points  
   - MediaPipe: 468 points  

2. **Eye-region cropping** using Dlib landmarks 36~47 + padding  

3. **Image transforms**:
   - Resize to 224×224  
   - Normalize with ImageNet mean/std  

---

### 3. Model Architecture

**MultiModalBlinkModel**

- **Eye-crop CNN branch**: ResNet18 → 512-dim feature  
- **Landmark branch**: Fully connected layers → 512-dim feature  
- **Fusion**: Concatenate CNN + landmark → 256-dim hidden → 2-class output (Open/Closed)  
- **Loss**: Weighted CrossEntropyLoss  

---

### 4. Inference Pipeline

1. Read webcam frame → flip horizontally → convert to grayscale/RGB  
2. Detect face:
   - Dlib: bounding box + 68-point landmarks  
   - MediaPipe: 468-point landmarks  

3. Crop eye region, compute CNN features  

4. Compute feature vector from landmarks (normalized by frame width/height)  

5. **Blink Prediction**:
   - Forward through MultiModalBlinkModel  
   - Class: 0 → Closed, 1 → Open  

6. **Gaze Estimation**:
   - Compute relative pupil x-position from landmarks  
   - Map to range [-1, 1]  

7. **Head Pose Estimation (PnP)**:
   - Use 6 key 2D points + canonical 3D face model  
   - Compute rotation vector → convert to Euler angles (Yaw, Pitch)  

8. Annotate frame with:
   - Blink status, Blink count  
   - Pupil position  
   - Head Yaw/Pitch  

---

### 5. Run Example

```bash
python run_blink_gaze_headpose.py
