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
# 🔴 NEW: 5.5️⃣ Head Pose (PnP) 설정을 위한 변수
# -------------------
# PnP를 위한 3D 얼굴 모델 포인트 (MediaPipe 랜드마크 기준)
# 스케일은 중요하지 않으며, 상대적인 위치가 중요함.
model_points = np.array([
    (0.0, 0.0, 0.0),             # 1. 코 끝 (Nose tip)
    (0.0, -330.0, -65.0),        # 152. 턱 (Chin)
    (-225.0, 170.0, -135.0),     # 33. 왼쪽 눈 왼쪽 끝 (Left eye left corner)
    (225.0, 170.0, -135.0),      # 263. 오른쪽 눈 오른쪽 끝 (Right eye right corner)
    (-150.0, -150.0, -125.0),    # 61. 왼쪽 입 끝 (Left mouth corner)
    (150.0, -150.0, -125.0)      # 291. 오른쪽 입 끝 (Right mouth corner)
])

# 카메라 매트릭스 (웹캠 크기에 따라 루프 진입 전 설정)
camera_matrix = np.zeros((3,3))
dist_coeffs = np.zeros((4, 1)) # 렌즈 왜곡 없다고 가정

# -------------------
# 6️⃣ 웹캠 실행
# -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 🔴 NEW: PnP를 위해 프레임 크기를 먼저 읽어와서 카메라 매트릭스 설정
ret, frame = cap.read()
if not ret:
    print("오류: 웹캠에서 첫 프레임을 읽을 수 없습니다.")
    cap.release()
    exit()
    
h, w, _ = frame.shape
FOCAL_LENGTH_ESTIMATE = w # 간단한 추정 (일반적으로 w와 비슷)
camera_matrix = np.array([
    [FOCAL_LENGTH_ESTIMATE, 0, w / 2],
    [0, FOCAL_LENGTH_ESTIMATE, h / 2],
    [0, 0, 1]
], dtype="double")
print(f"Frame (h, w) = ({h}, {w}). PnP용 Camera matrix 초기화 완료.")


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
    
    # 🔴 NEW: 헤드 포즈 변수 초기화
    head_pitch = 0.0
    head_yaw = 0.0
    
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
        # h, w 는 루프 밖에서 이미 정의됨
        dlib_coords = []
        for p in shape.parts():
            dlib_coords.extend([p.x / w, p.y / h]) 
        dlib_f = np.array(dlib_coords)

        mp_f = np.zeros(468 * 3)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark 
            mp_coords = []
            
            # 🔴 NEW: Head Pose (PnP) 계산 (MediaPipe 랜드마크 사용)
            try:
                # PnP에 사용할 2D 이미지 포인트
                image_points = np.array([
                    (landmarks[1].x * w, landmarks[1].y * h),    # 1. Nose
                    (landmarks[152].x * w, landmarks[152].y * h), # 152. Chin
                    (landmarks[33].x * w, landmarks[33].y * h),   # 33. Left eye corner
                    (landmarks[263].x * w, landmarks[263].y * h), # 263. Right eye corner
                    (landmarks[61].x * w, landmarks[61].y * h),   # 61. Left mouth corner
                    (landmarks[291].x * w, landmarks[291].y * h)  # 291. Right mouth corner
                ], dtype="double")
                
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points, 
                    image_points, 
                    camera_matrix, 
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE # (cv2.SOLVEPNP_SQPNP or cv2.SOLVEPNP_ITERATIVE)
                )
                
                # 회전 벡터를 Euler 각도로 변환 (Yaw, Pitch, Roll)
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                singular = sy < 1e-6
                
                if not singular:
                    head_pitch = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
                    head_yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
                    # head_roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
                else:
                    head_pitch = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
                    head_yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1]) * 180 / np.pi
                    # head_roll = 0.0
                
                # 🔴 NEW: PnP 결과 (얼굴 방향) 시각화 (보라색 선)
                (nose_end_point2D, _) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 500.0)]), # 코 끝(0,0,0)에서 Z축(정면)으로 500mm
                    rotation_vector, 
                    translation_vector, 
                    camera_matrix, 
                    dist_coeffs
                )
                p1 = (int(image_points[0][0]), int(image_points[0][1])) # 코 끝
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.arrowedLine(frame, p1, p2, (255, 0, 255), 3) # 보라색
                    
            except Exception as e:
                # print(f"Head pose PnP error: {e}")
                head_pitch, head_yaw = 0.0, 0.0
            
            # (기존 코드) 특징 벡터 준비
            for i in range(468):
                lm = landmarks[i]
                mp_coords.extend([lm.x, lm.y, lm.z])
            mp_f = np.array(mp_coords)

            # --- 🔴 3. 시선 추정 (MediaPipe 동공) - 숫자 매핑 (기존 유지) ---
            try:
                outer_corner_x = landmarks[33].x
                inner_corner_x = landmarks[133].x
                pupil_x = landmarks[473].x
                
                eye_width = abs(inner_corner_x - outer_corner_x)
                if eye_width > 0: 
                    pupil_pos = (pupil_x - outer_corner_x) / eye_width
                    gaze_value = (pupil_pos - 0.5) * 2.0
                    gaze_text = f"{gaze_value:.2f}"
                    
            except Exception as e:
                gaze_text = "Error"
                
        features = np.concatenate([dlib_f, mp_f]) 
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
    else:
        face_detected = False

    # --- 4. 모델 추론 (깜빡임) ---
    pred = 1 
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
            # print(f"Blink! (Total: {blink_count})") # 콘솔 출력 줄임

    # --- 6. 화면 표시 (수정됨) ---
    if not face_detected:
        status_text = "Face Not Detected"
        status_color = (0, 0, 255)
        gaze_text = "N/A"
        head_yaw, head_pitch = 0.0, 0.0 # N/A로 표시되도록 리셋
    else:
        status_text = "Closed" if pred == 0 else "Open"
        status_color = (0, 0, 255) if pred == 0 else (0, 255, 0)
    
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(frame, f"Blink Count: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # ### 🔴 NEW: 시선 텍스트 + 헤드 포즈 각도 표시 ###
    gaze_display_color = (128, 128, 128) # 기본값 (회색)
    
    if pred == 1 and face_detected: 
        # 눈을 떴고 얼굴이 감지되었을 때만 시선/각도 표시
        gaze_display_color = (0, 255, 255) # 노란색
        
        # 1. 동공 기준 (상대 위치)
        cv2.putText(frame, f"Pupil Pos: {gaze_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_display_color, 2)
        
        # 2. 얼굴 각도 (PnP)
        # Yaw (좌우), Pitch (상하)
        cv2.putText(frame, f"Head Yaw (L/R): {head_yaw:.1f} deg", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_display_color, 2)
        cv2.putText(frame, f"Head Pitch (U/D): {head_pitch:.1f} deg", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_display_color, 2)
    
    else:
        # 얼굴이 없거나 눈을 감았을 때
        cv2.putText(frame, f"Pupil Pos: N/A", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_display_color, 2)
        cv2.putText(frame, f"Head Yaw (L/R): N/A", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_display_color, 2)
        cv2.putText(frame, f"Head Pitch (U/D): N/A", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_display_color, 2)


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
