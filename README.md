# Eye Blink & Drowsiness Detection using Dlib, MediaPipe, and Transfer Learning  
![Python](https://img.shields.io/badge/python-3.x-blue.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-Transfer_Learning-orange.svg)  
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A deep learning–based **eye blink and drowsiness detection system**  
that leverages **open datasets**, **Dlib**, and **MediaPipe** for robust eye tracking  
and uses **transfer learning** for accurate eye state classification and drowsiness estimation.

---

## 🚀 Overview  
This project enhances traditional Dlib-based drowsiness detection by combining:  
- **MediaPipe Face Mesh** for high-speed facial landmark extraction  
- **Dlib** as a fallback for robust face detection under difficult conditions  
- **Transfer Learning** on open eye image datasets (e.g., CEW, ZJU, RT-BENE)  
- **Additional validation on drowsiness datasets** to assess fatigue levels  

The trained model predicts **eye open/closed state** and performs **blink-based drowsiness detection** in real time.

---

## ✨ Features  
- 🔹 Multi-source facial landmark detection (MediaPipe + Dlib)  
- 🔹 Eye-region cropping & normalization  
- 🔹 CNN transfer learning for eye-state classification  
- 🔹 Drowsiness estimation based on blink frequency and eye closure duration  
- 🔹 Real-time webcam inference with alert system  

---

## 🧠 Methodology  

### 1. Data  
The model was trained and validated using several open datasets:  
| Purpose | Dataset | Source | Description |
|----------|----------|--------|-------------|
| Training | [CEW Dataset](https://www.kaggle.com/datasets/ahamedfarouk/cew-dataset) | Kaggle | Closed Eyes in the Wild – labeled open/closed eye images. |
| Validation | [Drowness Detection Dataset](https://www.kaggle.com/datasets/norannabil/drowness-detection) | Kaggle | Used to evaluate model performance on real drowsy subjects. |

These datasets collectively provide a wide range of lighting conditions, facial orientations, and drowsiness levels.

---

### 2. Preprocessing  
- MediaPipe extracts 468 face landmarks.  
- Eye regions are cropped using landmark coordinates.  
- Images are resized and normalized to 112×112.  
- Data augmentation includes brightness, rotation, and occlusion.  

---

### 3. Model Architecture  
- Backbone: **MobileNetV2 / EfficientNet-B0 / ResNet-18** (transfer learning)  
- Output: Binary (Open / Closed) or probability-based Eye-State score  
- Loss: Binary Cross-Entropy (BCE)  
- Optimizer: AdamW + CosineAnnealingLR or ReduceLROnPlateau  
- Optional: Temporal module (LSTM / Transformer) for blink-sequence modeling  

---

### 4. Drowsiness Validation  
To assess fatigue detection accuracy, validation was performed using the  
**[Drowness Detection Dataset](https://www.kaggle.com/datasets/norannabil/drowness-detection)** from Kaggle.  
This dataset includes facial images labeled with different **drowsiness levels**.  

During validation:
- The trained blink model was used to extract eye-state probabilities per frame.  
- Features such as **blink frequency**, **PERCLOS** (Percentage of Eye Closure),  
  and **average closure duration** were computed.  
- A separate lightweight classifier estimated the drowsiness score.  

```python
# Example: validation snippet
from evaluate_drowsiness import evaluate_drowsiness

metrics = evaluate_drowsiness(model, drowness_dataset_path="./datasets/drowness")
print(metrics)
# Example output (illustrative only):
# {'accuracy': 0.93, 'f1_drowsy': 0.91, 'pearson_r': 0.88}
