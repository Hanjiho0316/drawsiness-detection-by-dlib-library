import cv2
import dlib
import numpy as np
import mediapipe as mp

# =============================
# 1. dlib 모델 및 mediapipe 초기화
# =============================
predictor_path = r"\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# =============================
# 2. dlib 랜드마크 추출 함수
# =============================
def extract_dlib_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    # 얼굴이 안 잡히면 크기를 줄여 재시도
    if len(rects) == 0:
        resized = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        rects = detector(resized, 1)
        if len(rects) == 0:
            return None

    shape = predictor(gray, rects[0])
    coords = np.array([[p.x, p.y] for p in shape.parts()])
    return coords

# =============================
# 3. MediaPipe 랜드마크 추출 함수
# =============================
def extract_mediapipe_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    h, w, _ = img.shape
    face_landmarks = result.multi_face_landmarks[0]
    coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark])
    return coords

# =============================
# 4. 통합 랜드마크 추출 및 시각화
# =============================
def visualize_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("⚠️ 이미지를 불러올 수 없습니다:", image_path)
        return

    dlib_coords = extract_dlib_landmarks(img)
    mp_coords = extract_mediapipe_landmarks(img)

    if dlib_coords is not None:
        for (x, y) in dlib_coords:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    else:
        print("⚠️ Dlib이 얼굴을 찾지 못했습니다.")

    if mp_coords is not None:
        for (x, y) in mp_coords[::10]:  # landmark 너무 많으므로 10개마다 표시
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
    else:
        print("⚠️ MediaPipe가 얼굴을 찾지 못했습니다.")

    # 좌표 병합 (dlib + mediapipe)
    merged = None
    if dlib_coords is not None and mp_coords is not None:
        merged = np.concatenate([dlib_coords.flatten(), mp_coords.flatten()])

    # 시각화
    cv2.imshow("Dlib (Green) + MediaPipe (Blue) Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if merged is not None:
        print(f"✅ 병합된 feature shape: {merged.shape}")
    else:
        print("❌ 병합 불가 (하나 이상 탐지 실패)")

# =============================
# 5. 실행
# =============================
if __name__ == "__main__":
    image_path = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\sample_face.jpg"
    visualize_landmarks(image_path)
