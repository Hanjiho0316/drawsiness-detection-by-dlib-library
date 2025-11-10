import cv2
import mediapipe as mp

# ------------------- 설정 -------------------
image_path = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\sample_face.jpg"  # 👈 분석할 이미지 경로
save_path = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\mediapipe_eye_landmarks.jpg"

# ------------------- MediaPipe 초기화 -------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 눈 영역 인덱스 (468개 랜드마크 중 눈 부분만)
LEFT_EYE_IDX = list(range(33, 133))
RIGHT_EYE_IDX = list(range(362, 463))

# ------------------- 이미지 읽기 -------------------
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ Image not found: {image_path}")
h, w, _ = image.shape
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ------------------- FaceMesh 처리 -------------------
results = face_mesh.process(rgb)

if not results.multi_face_landmarks:
    print("❌ No face detected.")
else:
    face_landmarks = results.multi_face_landmarks[0]

    # 왼쪽 눈 랜드마크 찍기 (파란색)
    for idx in LEFT_EYE_IDX:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

    # 오른쪽 눈 랜드마크 찍기 (빨간색)
    for idx in RIGHT_EYE_IDX:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # 결과 저장 및 표시
    cv2.imwrite(save_path, image)
    print(f"✅ Eye landmarks saved to: {save_path}")

    cv2.imshow("Eye Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
