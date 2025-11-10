# MultiModal Eye Blink Detection using Dlib, MediaPipe, and Transfer Learning  
![Python](https://img.shields.io/badge/python-3.x-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-Transfer_Learning-orange.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green.svg)  
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A **multimodal eye blink detection system** using **eye-region images** and **facial landmarks**.  
The system leverages **Dlib**, **MediaPipe**, and **CNN-based transfer learning** to predict **open/closed eye states** in real time.

---

## 🚀 Overview  

This project improves traditional Dlib-based blink detection by combining:  
- **MediaPipe Face Mesh** for 468-point high-speed facial landmark extraction  
- **Dlib 68-point landmarks** for robustness in difficult conditions  
- **Eye-region cropping** for focused CNN feature extraction  
- **Multimodal fusion**: combining eye-crop CNN features + full-face landmark features  
- **Weighted CrossEntropyLoss** to handle class imbalance  

The model predicts **blink status** (open/closed) and can be extended for **drowsiness detection**.

---

## ✨ Features  

- 🔹 Multimodal input: eye-crop images + full-face landmarks  
- 🔹 Transfer learning with **ResNet18** backbone  
- 🔹 Weighted loss for imbalanced datasets (Open/Closed Eyes)  
- 🔹 Real-time webcam inference (extension possible)  
- 🔹 Training logs saved as CSV for easy monitoring  

---

## 🧠 Methodology  

### 1. Dataset  
The model is trained primarily on the **CEW Dataset**:

| Dataset | Source | Description |
|---------|--------|-------------|
| CEW | [Kaggle](https://www.kaggle.com/datasets/ahamedfarouk/cew-dataset) | Labeled open/closed eye images for training the eye-state classifier |
| Validation | [Drowness Detection Dataset](https://www.kaggle.com/datasets/norannabil/drowness-detection) | Optional for evaluating blink-based drowsiness |

- Open eyes labeled as `1`, Closed eyes labeled as `0`  
- Class imbalance handled by **weighted CrossEntropyLoss**

---

### 2. Preprocessing  
1. **Facial landmarks extraction**:  
   - Dlib: 68-point landmarks  
   - MediaPipe: 468-point landmarks  
2. **Eye-region cropping** using landmarks 36~47 + padding  
3. **Image transforms**:
   - Training: resize, horizontal flip, color jitter, normalization  
   - Validation: resize + normalization only  

---

### 3. Model Architecture  

**MultiModalBlinkModel**  

- **Eye-crop CNN branch**: ResNet18 (pretrained) → 512-dim features  
- **Landmark branch**: Fully connected layers → 512-dim features  
- **Fusion**: Concatenate CNN + landmark features → 256-dim hidden → 2-class output  
- **Loss**: Weighted CrossEntropyLoss (handles open/closed imbalance)  

---

### 4. Training Loop  

- Epochs: 30  
- Batch size: 8  
- Optimizer: Adam (lr=1e-4)  
- Device: GPU if available, otherwise CPU  
- Logs saved to CSV (`train_multitask_log.csv`) with columns:  
  `epoch`, `train_blink_loss`, `val_blink_loss`, `val_acc`  

Example log record:

```csv
epoch,train_blink_loss,val_blink_loss,val_acc
1,0.5123,0.4789,0.8421
2,0.4312,0.4021,0.8679
...
