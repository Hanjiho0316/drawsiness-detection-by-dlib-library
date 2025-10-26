import os
import cv2
import dlib
import torch
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# -------------------
# 1️⃣ 설정
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICTOR_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\shape_predictor_68_face_landmarks.dat"
cew_root = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\CEW"

SAVE_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\best_multitask_model.pth"
LOG_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\train_multitask_log.csv"

LAMBDA_EYE = 0.01  # Eye Loss 가중치
EPOCHS = 60

# -------------------
# 2️⃣ dlib + MediaPipe Feature Extractor
# -------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    if len(rects) == 0:
        return None, None

    shape = predictor(gray, rects[0])
    dlib_pts = np.array([[p.x, p.y] for p in shape.parts()]).flatten()  # (136,)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    if not results.multi_face_landmarks:
        return dlib_pts, None

    mp_pts = []
    for lm in results.multi_face_landmarks[0].landmark:
        mp_pts.extend([lm.x, lm.y, lm.z])
    mp_pts = np.array(mp_pts)
    return dlib_pts, mp_pts

# -------------------
# 3️⃣ Dataset Class
# -------------------
class EyeDataset(Dataset):
    def __init__(self, img_paths, labels, eye_bboxes=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.eye_bboxes = eye_bboxes  # (x1,y1,x2,y2) 좌표, 없으면 zeros
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        dlib_f, mp_f = extract_features(img)
        if dlib_f is None:
            dlib_f = np.zeros(68*2)
        if mp_f is None:
            mp_f = np.zeros(468*3)

        features = np.concatenate([dlib_f, mp_f])

        # Eye bbox 정규화
        if self.eye_bboxes is not None:
            eye_bbox = self.eye_bboxes[idx] / 224.0  # 0~1
        else:
            eye_bbox = np.zeros(4, dtype=np.float32)

        if self.transform:
            img = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(eye_bbox, dtype=torch.float32)

# -------------------
# 4️⃣ Multimodal Model
# -------------------
class MultiModalBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2 + 468*3):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
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
            nn.Linear(256, 2)
        )

        self.eye_fc = nn.Sequential(
            nn.Linear(512 + img_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # x1,y1,x2,y2
        )

    def forward(self, img, features):
        img_feat = self.cnn(img)
        feat = self.feature_fc(features)
        combined = torch.cat((img_feat, feat), dim=1)
        blink_out = self.blink_fc(combined)
        eye_out = self.eye_fc(combined)
        return blink_out, eye_out

# -------------------
# 5️⃣ 이미지/라벨 준비
# -------------------
open_imgs = [os.path.join(cew_root, "OpenEye", f) for f in os.listdir(os.path.join(cew_root, "OpenEye")) if f.lower().endswith(('.jpg','.png'))]
closed_imgs = [os.path.join(cew_root, "ClosedEye", f) for f in os.listdir(os.path.join(cew_root, "ClosedEye")) if f.lower().endswith(('.jpg','.png'))]

img_paths = open_imgs + closed_imgs
labels = [1]*len(open_imgs) + [0]*len(closed_imgs)

# eye_bbox는 예시로 중앙 정사각형 (실제 데이터셋에 맞춰 넣으면 더 정확)
eye_bboxes = np.array([[60,60,160,120]]*len(img_paths), dtype=np.float32)

train_paths, val_paths, train_labels, val_labels, train_bboxes, val_bboxes = train_test_split(
    img_paths, labels, eye_bboxes, test_size=0.2, random_state=42
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = EyeDataset(train_paths, train_labels, train_bboxes, transform)
val_dataset = EyeDataset(val_paths, val_labels, val_bboxes, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# -------------------
# 6️⃣ 학습 루프
# -------------------
model = MultiModalBlinkModel().to(DEVICE)
criterion_blink = nn.CrossEntropyLoss()
criterion_eye = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')
log_records = []

for epoch in range(1, EPOCHS+1):
    model.train()
    total_blink_loss, total_eye_loss = 0, 0
    for img, feat, label, eye_bbox in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        img, feat, label, eye_bbox = img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE), eye_bbox.to(DEVICE)
        optimizer.zero_grad()
        blink_out, eye_out = model(img, feat)
        blink_loss = criterion_blink(blink_out, label)
        eye_loss = criterion_eye(eye_out, eye_bbox)
        loss = blink_loss + LAMBDA_EYE * eye_loss
        loss.backward()
        optimizer.step()
        total_blink_loss += blink_loss.item()
        total_eye_loss += eye_loss.item()
    total_blink_loss /= len(train_loader)
    total_eye_loss /= len(train_loader)

    # Validation
    model.eval()
    val_blink_loss, val_eye_loss = 0,0
    correct, total = 0,0
    with torch.no_grad():
        for img, feat, label, eye_bbox in val_loader:
            img, feat, label, eye_bbox = img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE), eye_bbox.to(DEVICE)
            blink_out, eye_out = model(img, feat)
            blink_loss = criterion_blink(blink_out, label)
            eye_loss = criterion_eye(eye_out, eye_bbox)
            val_blink_loss += blink_loss.item()
            val_eye_loss += eye_loss.item()
            pred = torch.argmax(blink_out, dim=1)
            correct += (pred==label).sum().item()
            total += label.size(0)
    val_blink_loss /= len(val_loader)
    val_eye_loss /= len(val_loader)
    val_total_loss = val_blink_loss + LAMBDA_EYE * val_eye_loss
    val_acc = correct / total

    print(f"Epoch {epoch}/{EPOCHS} - Train Blink: {total_blink_loss:.4f}, Eye: {total_eye_loss:.4f} | Val Blink: {val_blink_loss:.4f}, Eye: {val_eye_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Best model 저장
    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  -> Best model saved with val loss {best_val_loss:.4f}")

    # 로그 기록
    log_records.append({
        "epoch": epoch,
        "train_blink_loss": total_blink_loss,
        "train_eye_loss": total_eye_loss,
        "val_blink_loss": val_blink_loss,
        "val_eye_loss": val_eye_loss,
        "val_acc": val_acc
    })

# 로그 CSV 저장
df_log = pd.DataFrame(log_records)
df_log.to_csv(LOG_PATH, index=False)
print(f"Training log saved to {LOG_PATH}")
