import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import dlib
import mediapipe as mp

# -------------------
# 1️⃣ 설정
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICTOR_PATH = "/Users/hanjiho/Desktop/eye detect/eye_blink_detector-master/face recognition/shape_predictor_68_face_landmarks.dat"
SAVE_PATH = "/Users/hanjiho/Desktop/eye detect/eye_blink_detector-master/face recognition/best_multitask_model.pth"

EYE_CROP_PADDING = 20
# (임계값 대신 숫자 매핑을 사용하므로 임계값 설정 제거)
# GAZE_THRESHOLD_LEFT = 0.4
# GAZE_THRESHOLD_RIGHT = 0.6

# -------------------
# 2️⃣ dlib + MediaPipe 초기화
# -------------------
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError as e:
    print(f"dlib predictor 로드 오류: {e}")
    exit()

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True # 동공 좌표(473)를 위해 True
)

# -------------------
# 3️⃣ Multimodal Model (훈련 코드(R20)와 동일)
# -------------------
class MultiModalBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2 + 468*3): # 1540
        super().__init__()
        base_model = models.resnet18(pretrained=False) 
        base_model.fc = nn.Identity()
        self.cnn = base_model 

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 512), # 1540 -> 512
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.blink_fc = nn.Sequential(
            nn.Linear(512 + img_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img, features):
        img_feat = self.cnn(img)
        feat = self.feature_fc(features)
        combined = torch.cat((img_feat, feat), dim=1)
        blink_out = self.blink_fc(combined)
        return blink_out

# -------------------
# 4️⃣ 모델 로드
# -------------------
feature_dim = 68*2 + 468*3 # 1540
model = MultiModalBlinkModel(feature_dim=feature_dim).to(DEVICE)
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval() 
print(f"모델 로드 완료: {SAVE_PATH}")

# -------------------
# 5️⃣ Transform 정의
# -------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# -------------------
# 6️⃣ 웹캠 실행
# -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

blink_flag = False
blink_count = 0

print("웹캠 실행 중... (ESC 키를 누르면 종료됩니다)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("오류: 프레임을 읽을 수 없습니다.")
        break
    
    frame = cv2.flip(frame, 1) 
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = mp_face.process(rgb_frame)
    rects = detector(gray_frame)
    
    face_detected = False
    gaze_text = "N/A" # 시선 기본값 (문자열로 유지)
    
    img_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(DEVICE)
    features_tensor = torch.zeros((1, feature_dim), dtype=torch.float32).to(DEVICE)
    
    if len(rects) > 0:
        face_detected = True
        shape = predictor(gray_frame, rects[0])
        
        # --- 1. 모델 입력 (이미지) & 눈 위치 시각화 (dlib 기반) ---
        eye_coords_x = []
        eye_coords_y = []
        for i in range(36, 48):
            eye_coords_x.append(shape.part(i).x)
            eye_coords_y.append(shape.part(i).y)
        
        x_min = max(0, min(eye_coords_x) - EYE_CROP_PADDING)
        x_max = min(frame.shape[1], max(eye_coords_x) + EYE_CROP_PADDING)
        y_min = max(0, min(eye_coords_y) - EYE_CROP_PADDING)
        y_max = min(frame.shape[0], max(eye_coords_y) + EYE_CROP_PADDING)
        
        if x_max > x_min and y_max > y_min:
            eye_crop_img = frame[y_min:y_max, x_min:x_max]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            rgb_crop = cv2.cvtColor(eye_crop_img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(rgb_crop).unsqueeze(0).to(DEVICE)
        else:
            face_detected = False

        # --- 2. 모델 입력 (특징) 준비 (dlib + MediaPipe) ---
        h, w, _ = frame.shape
        dlib_coords = []
        for p in shape.parts():
            dlib_coords.extend([p.x / w, p.y / h]) 
        dlib_f = np.array(dlib_coords)

        mp_f = np.zeros(468 * 3)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark 
            mp_coords = []
            
            for i in range(468):
                lm = landmarks[i]
                mp_coords.extend([lm.x, lm.y, lm.z])
            mp_f = np.array(mp_coords)

            # --- 🔴 3. 시선 추정 (MediaPipe 동공) - 숫자 매핑으로 수정 ---
            try:
                outer_corner_x = landmarks[33].x
                inner_corner_x = landmarks[133].x
                pupil_x = landmarks[473].x
                
                eye_width = abs(inner_corner_x - outer_corner_x)
                if eye_width > 0: 
                    # 0.0 ~ 1.0 사이의 상대 위치
                    pupil_pos = (pupil_x - outer_corner_x) / eye_width

                    # 0.0~1.0 범위를 -1.0~1.0 범위로 매핑 (0.5가 0.0이 됨)
                    gaze_value = (pupil_pos - 0.5) * 2.0
                    
                    # (참고) 프레임이 반전되었으므로,
                    # gaze_value < 0 -> 사용자가 왼쪽을 봄
                    # gaze_value > 0 -> 사용자가 오른쪽을 봄
                    
                    gaze_text = f"{gaze_value:.2f}" # 숫자를 문자열로 포맷팅
                    
            except Exception as e:
                gaze_text = "Error"
                
        features = np.concatenate([dlib_f, mp_f]) 
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
    else:
        face_detected = False

    # --- 4. 모델 추론 (깜빡임) ---
    pred = 1 # 기본값 'Open'
    if face_detected:
        with torch.no_grad():
            blink_out = model(img_tensor, features_tensor)
            pred = torch.argmax(blink_out, dim=1).item() 
    else:
        gaze_text = "N/A" 

    # --- 5. 깜빡임 카운트 ---
    if pred == 0: 
        if not blink_flag:
            blink_flag = True
    else: 
        if blink_flag:
            blink_flag = False
            blink_count += 1
            print(f"Blink! (Total: {blink_count})")

    # --- 6. 화면 표시 (수정됨) ---
    if not face_detected:
        status_text = "Face Not Detected"
        status_color = (0, 0, 255)
    else:
        status_text = "Closed" if pred == 0 else "Open"
        status_color = (0, 0, 255) if pred == 0 else (0, 255, 0)
    
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(frame, f"Blink Count: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # ### 🔴 수정: 시선 텍스트가 숫자로 표시됨 ###
    if pred == 1 and face_detected: 
        cv2.putText(frame, f"Gaze: {gaze_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Gaze: N/A", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

    cv2.imshow("Blink Detection + Gaze (ESC to exit)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
        break

# -------------------
# 7️⃣ 종료
# -------------------
cap.release()
mp_face.close()
cv2.destroyAllWindows()
print(f"Final Blink Count: {blink_count}")
