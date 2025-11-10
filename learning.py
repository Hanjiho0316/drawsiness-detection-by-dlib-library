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
LOG_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognitiontrain_multitask_log.csv"

EPOCHS = 30
EYE_CROP_PADDING = 20 # 눈 크롭 시 여백

# -------------------
# 2️⃣ dlib + MediaPipe Feature Extractor
# -------------------
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError:
    print(f"Error: dlib predictor '{PREDICTOR_PATH}'를 찾을 수 없습니다.")
    exit()
    
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 이 함수는 '얼굴 전체' 이미지를 받아 '얼굴 전체' 랜드마크를 반환합니다.
def extract_features(image):
    if image is None:
        return np.zeros(68*2), np.zeros(468*3)
        
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    
    dlib_pts = np.zeros(68 * 2) 
    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        dlib_coords = []
        for p in shape.parts():
            dlib_coords.extend([p.x / w, p.y / h]) 
        dlib_pts = np.array(dlib_coords)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    
    mp_pts = np.zeros(468 * 3) 
    if results.multi_face_landmarks:
        mp_coords = []
        for lm in results.multi_face_landmarks[0].landmark:
            mp_coords.extend([lm.x, lm.y, lm.z])
        mp_pts = np.array(mp_coords)

    return dlib_pts, mp_pts

# -------------------
# 3️⃣ Dataset Class (🔴 핵심 수정)
# -------------------
class EyeDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        # dlib은 __getitem__ 안에서 매번 초기화하면 매우 느리므로, 여기서 미리 로드
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # 1. '얼굴 전체' 이미지 로드
        img = cv2.imread(img_path) 
        if img is None:
            print(f"Warning: Cannot read image, returning zeros: {img_path}")
            dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_feat = torch.zeros((68*2 + 468*3), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_img, dummy_feat, dummy_label

        # 2. 입력 1: '얼굴 전체' 랜드마크 추출
        dlib_f, mp_f = extract_features(img)
        features = np.concatenate([dlib_f, mp_f])

        # 3. 입력 2: '눈 영역' 크롭
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray)
        
        eye_crop_img = img # 기본값 (얼굴 감지 실패 시)
        
        if len(rects) > 0:
            shape = self.predictor(gray, rects[0])
            eye_coords_x = []
            eye_coords_y = []
            for i in range(36, 48): # 랜드마크 36~47 (양쪽 눈)
                eye_coords_x.append(shape.part(i).x)
                eye_coords_y.append(shape.part(i).y)
            
            x_min = max(0, min(eye_coords_x) - EYE_CROP_PADDING)
            x_max = min(img.shape[1], max(eye_coords_x) + EYE_CROP_PADDING)
            y_min = max(0, min(eye_coords_y) - EYE_CROP_PADDING)
            y_max = min(img.shape[0], max(eye_coords_y) + EYE_CROP_PADDING)
            
            if x_max > x_min and y_max > y_min:
                eye_crop_img = img[y_min:y_max, x_min:x_max]

        # 4. '눈 크롭' 이미지 Transform
        if self.transform:
            # BGR -> RGB 변환 후 transform 적용
            eye_crop_img = self.transform(cv2.cvtColor(eye_crop_img, cv2.COLOR_BGR2RGB))
        
        # (이미지, 랜드마크, 라벨) 반환
        return eye_crop_img, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -------------------
# 4️⃣ Multimodal Model (🔴 핵심 수정)
# -------------------
class MultiModalBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2 + 468*3):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Identity()
        self.cnn = base_model # (입력: 눈 크롭 이미지)

        self.feature_fc = nn.Sequential( # (입력: 얼굴 전체 랜드마크)
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.blink_fc = nn.Sequential(
            nn.Linear(512 + img_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # ### 🔴 수정된 부분: '가짜' 목표였던 eye_fc 제거 ###
        # self.eye_fc = nn.Sequential(...) 

    def forward(self, img, features):
        img_feat = self.cnn(img)
        feat = self.feature_fc(features)
        combined = torch.cat((img_feat, feat), dim=1)
        
        blink_out = self.blink_fc(combined)
        
        # ### 🔴 수정된 부분: eye_out 반환 제거 ###
        return blink_out 

# ----------------------------------------------------
# 💥💥💥 메인 실행 블록 (if __name__ == '__main__') 💥💥💥
# ----------------------------------------------------
if __name__ == '__main__':

    # -------------------
    # 5️⃣ 이미지/라벨 준비 (가중치 계산 추가)
    # -------------------
    open_imgs = [os.path.join(cew_root, "OpenEye", f) for f in os.listdir(os.path.join(cew_root, "OpenEye")) if f.lower().endswith(('.jpg','.png'))]
    closed_imgs = [os.path.join(cew_root, "ClosedEye", f) for f in os.listdir(os.path.join(cew_root, "ClosedEye")) if f.lower().endswith(('.jpg','.png'))]

    img_paths = open_imgs + closed_imgs
    labels = [1]*len(open_imgs) + [0]*len(closed_imgs) # 1: Open, 0: Closed

    print(f"Total Images: {len(img_paths)}")
    print(f"  Open Eyes (Label 1): {len(open_imgs)}")
    print(f"  Closed Eyes (Label 0): {len(closed_imgs)}")

    # ### 데이터 불균형 해소를 위한 가중치 계산 ###
    count_0 = len(closed_imgs)
    count_1 = len(open_imgs)
    
    class_weights = None
    if count_0 > 0 and count_1 > 0:
        total = count_0 + count_1
        weight_0 = total / (2.0 * count_0)
        weight_1 = total / (2.0 * count_1)
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(DEVICE)
        print(f"Applying weights - 0(Closed): {weight_0:.2f}, 1(Open): {weight_1:.2f}")
    else:
        print("경고: 한 클래스의 데이터가 0개입니다. 가중치를 적용하지 않습니다.")

    # (eye_bboxes 제거)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 훈련용 Transform (Augmentation 포함)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) 
    ])

    # 검증용 Transform (Augmentation 없음)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # (이전 코드의 0.255 오타 수정 -> 0.225)
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) 
    ])

    # ### 🔴 수정된 부분: EyeDataset에 eye_bboxes 인자 제거 ###
    train_dataset = EyeDataset(train_paths, train_labels, transform)
    val_dataset = EyeDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # -------------------
    # 6️⃣ 학습 루프 (🔴 핵심 수정)
    # -------------------
    model = MultiModalBlinkModel(feature_dim=68*2 + 468*3).to(DEVICE)
    
    # ### Loss에 가중치(weights) 적용 ###
    criterion_blink = nn.CrossEntropyLoss(weight=class_weights)
    
    # (criterion_eye 제거)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    log_records = []

    print(f"Training started on {DEVICE}...")

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_blink_loss = 0
        # ### 🔴 수정된 부분: eye_bbox 반환 X ###
        for eye_crop_img, feat, label in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            eye_crop_img, feat, label = eye_crop_img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            
            # ### 🔴 수정된 부분: blink_out만 반환 ###
            blink_out = model(eye_crop_img, feat)
            
            blink_loss = criterion_blink(blink_out, label)
            
            # (eye_loss 제거)
            loss = blink_loss 
            
            loss.backward()
            optimizer.step()
            
            total_blink_loss += blink_loss.item()
            
        total_blink_loss /= len(train_loader)

        # Validation
        model.eval()
        val_blink_loss = 0
        correct, total = 0,0
        with torch.no_grad():
            for eye_crop_img, feat, label in val_loader:
                eye_crop_img, feat, label = eye_crop_img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                
                blink_out = model(eye_crop_img, feat)
                
                blink_loss = criterion_blink(blink_out, label)
                val_blink_loss += blink_loss.item()
                
                pred = torch.argmax(blink_out, dim=1)
                correct += (pred==label).sum().item()
                total += label.size(0)
                
        val_blink_loss /= len(val_loader)
        val_acc = correct / total
        
        # (val_total_loss -> val_blink_loss)
        print(f"Epoch {epoch}/{EPOCHS} - Train Blink: {total_blink_loss:.4f} | Val Blink: {val_blink_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Best model 저장
        if val_blink_loss < best_val_loss:
            best_val_loss = val_blink_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Best model saved with val blink loss {best_val_loss:.4f}")

        # 로그 기록
        log_records.append({
            "epoch": epoch,
            "train_blink_loss": total_blink_loss,
            "val_blink_loss": val_blink_loss,
            "val_acc": val_acc
        })

    # 로그 CSV 저장
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(LOG_PATH, index=False)
    print(f"Training log saved to {LOG_PATH}")
    print("Training finished.")
