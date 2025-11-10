import os
import cv2
import dlib
import torch
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt  # ----- [추가됨]
import random  # ----- [추가됨]

# ------------------- 설정 -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREDICTOR_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\shape_predictor_68_face_landmarks.dat"
DATA_ROOT = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\data_cropped"
MODEL_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\best_blink_eye_model.pth"
SAVE_CSV = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\comparison_results.csv"
SAVE_PLOT = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\accuracy_comparison.png"  # ----- [추가됨]
SAVE_LANDMARK_IMG = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\landmark_comparison.jpg"  # ----- [추가됨]

# ------------------- 모델 정의 -------------------
class MultiTaskBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2 + 468*3, blink_classes=2, eye_classes=8):
        super().__init__()
        base_model = models.resnet18(pretrained=False)
        base_model.fc = nn.Identity()
        self.cnn = base_model

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.blink_fc = nn.Sequential(
            nn.Linear(512 + img_dim, 256),
            nn.ReLU(),
            nn.Linear(256, blink_classes)
        )

        self.eye_fc = nn.Sequential(
            nn.Linear(512 + img_dim, 256),
            nn.ReLU(),
            nn.Linear(256, eye_classes)
        )

    def forward(self, img, features):
        img_feat = self.cnn(img)
        feat = self.feature_fc(features)
        combined = torch.cat((img_feat, feat), dim=1)
        blink_out = self.blink_fc(combined)
        eye_out = self.eye_fc(combined)
        return blink_out, eye_out

# ------------------- 특징 추출 -------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    if len(rects) == 0:
        return None, None, None

    shape = predictor(gray, rects[0])
    dlib_pts = np.array([[p.x, p.y] for p in shape.parts()])

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    mp_pts = None
    if results.multi_face_landmarks:
        mp_pts = []
        for lm in results.multi_face_landmarks[0].landmark:
            mp_pts.extend([lm.x, lm.y, lm.z])
        mp_pts = np.array(mp_pts)

    left_eye_idx = [36, 37, 38, 39, 40, 41]
    right_eye_idx = [42, 43, 44, 45, 46, 47]
    left_eye_center = np.mean(dlib_pts[left_eye_idx], axis=0)
    right_eye_center = np.mean(dlib_pts[right_eye_idx], axis=0)

    return dlib_pts.flatten(), mp_pts, (left_eye_center, right_eye_center)

# EAR 계산
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def dlib_predict(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    if len(rects) == 0:
        return None, None

    shape = predictor(gray, rects[0])
    coords = np.array([[p.x, p.y] for p in shape.parts()])
    left_eye_idx = [36, 37, 38, 39, 40, 41]
    right_eye_idx = [42, 43, 44, 45, 46, 47]

    leftEAR = eye_aspect_ratio(coords[left_eye_idx])
    rightEAR = eye_aspect_ratio(coords[right_eye_idx])
    ear = (leftEAR + rightEAR) / 2.0

    left_center = np.mean(coords[left_eye_idx], axis=0)
    right_center = np.mean(coords[right_eye_idx], axis=0)
    return ear, (left_center, right_center)

# ------------------- 모델 로드 -------------------
model = MultiTaskBlinkModel(blink_classes=2, eye_classes=8).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def model_predict(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    dlib_f, mp_f, eyes = extract_features(img)
    if dlib_f is None:
        dlib_f = np.zeros(68*2)
    if mp_f is None:
        mp_f = np.zeros(468*3)
    features = np.concatenate([dlib_f, mp_f])
    img_tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        blink_out, eye_out = model(img_tensor, feat_tensor)
        blink_pred = torch.argmax(blink_out, dim=1).item()
        eye_pred = torch.argmax(eye_out, dim=1).item()

    return blink_pred, eye_pred, eyes

# ------------------- 데이터 루프 -------------------
categories = ["active", "sleep"]
records = []

for cls_idx, cls_name in enumerate(categories):
    folder = os.path.join(DATA_ROOT, cls_name)
    img_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png'))]

    for img_path in tqdm(img_files, desc=f"Testing {cls_name}"):
        ear, dlib_eyes = dlib_predict(img_path)
        blink_pred, eye_pred, model_eyes = model_predict(img_path)
        if ear is None or blink_pred is None:
            continue

        dlib_pred = 1 if ear >= 0.21 else 0

        records.append({
            "file": os.path.basename(img_path),
            "label": cls_idx,
            "dlib_pred": dlib_pred,
            "model_blink_pred": blink_pred,
            "model_eye_pred": eye_pred,
            "ear": ear,
            "dlib_left_x": dlib_eyes[0][0],
            "dlib_left_y": dlib_eyes[0][1],
            "model_left_x": model_eyes[0][0] if model_eyes else None,
            "model_left_y": model_eyes[0][1] if model_eyes else None
        })

# ------------------- 결과 분석 -------------------
df = pd.DataFrame(records)
dlib_acc = (df["label"] == df["dlib_pred"]).mean()
model_acc = (df["label"] == df["model_blink_pred"]).mean()

eye_diff = np.sqrt((df["dlib_left_x"] - df["model_left_x"])**2 + (df["dlib_left_y"] - df["model_left_y"])**2)
mean_eye_diff = np.nanmean(eye_diff)

print(f"\n✅ Dlib Accuracy (눈감김):  {dlib_acc:.4f}")
print(f"✅ Model Accuracy (눈감김): {model_acc:.4f}")
print(f"👁️ Mean Eye Position Difference: {mean_eye_diff:.2f} pixels")

df.to_csv(SAVE_CSV, index=False)
print(f"\nResults saved to: {SAVE_CSV}")

# ------------------- 시각화 -------------------
# 정확도 비교 막대그래프 저장 ----- [추가됨]
plt.figure(figsize=(5,5))
plt.bar(["Dlib", "Model"], [dlib_acc, model_acc], color=["skyblue", "salmon"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Blink Detection Accuracy Comparison")
for i, acc in enumerate([dlib_acc, model_acc]):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=12)
plt.savefig(SAVE_PLOT)
plt.close()
print(f"📊 Accuracy comparison plot saved to: {SAVE_PLOT}")

# ------------------- 랜드마크 시각화 -------------------
# 임의의 이미지 하나 선택 ----- [추가됨]
sample_cls = random.choice(categories)
sample_folder = os.path.join(DATA_ROOT, sample_cls)
sample_img = random.choice([os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg','.png'))])

img = cv2.imread(sample_img)
dlib_f, mp_f, eyes = extract_features(img)

if dlib_f is not None:
    dlib_pts = dlib_f.reshape(-1, 2)
    for (x, y) in dlib_pts:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)  # 초록색 점

if mp_f is not None:
    h, w, _ = img.shape
    mp_pts = mp_f.reshape(-1, 3)
    for (x, y, z) in mp_pts[::15]:  # 일부만 표시 (성능 위해)
        cv2.circle(img, (int(x * w), int(y * h)), 1, (0, 0, 255), -1)  # 빨간색 점

cv2.imwrite(SAVE_LANDMARK_IMG, img)
print(f"📷 Landmark visualization saved to: {SAVE_LANDMARK_IMG}")
print(f"Sample image used: {sample_img}")
