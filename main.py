import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# -----------------------
# EAR 계산 함수
# -----------------------
def eye_aspect_ratio(eye):
    # 수직 거리 계산
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 수평 거리 계산
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -----------------------
# 얼굴 랜드마크 불러오기
# -----------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 68 포인트 모델

# 눈 위치 인덱스
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# -----------------------
# 고개 각도 계산용 포인트
# -----------------------
# 3D 모델 포인트 (코끝, 턱, 눈, 입술)
model_points = np.array([
    (0.0, 0.0, 0.0),             # 코끝
    (0.0, -330.0, -65.0),        # 턱
    (-225.0, 170.0, -135.0),     # 왼쪽 눈
    (225.0, 170.0, -135.0),      # 오른쪽 눈
    (-150.0, -150.0, -125.0),    # 왼쪽 입술
    (150.0, -150.0, -125.0)      # 오른쪽 입술
])

# 카메라 매트릭스 설정 (웹캠 기준)
size = (640, 480)
focal_length = size[1]
center = (size[1]//2, size[0]//2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

# 초기화
blink_threshold = 0.25
blink_consec_frames = 3
counter = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # -----------------------
        # 눈 깜빡임 계산
        # -----------------------
        leftEye = shape[LEFT_EYE]
        rightEye = shape[RIGHT_EYE]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 눈 깜빡임 판단
        if ear < blink_threshold:
            counter += 1
        else:
            if counter >= blink_consec_frames:
                print("Blink detected!")
            counter = 0

        # 눈 표시
        for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # -----------------------
        # 고개 각도 계산
        # -----------------------
        image_points = np.array([
            shape[30],     # 코끝
            shape[8],      # 턱
            shape[36],     # 왼쪽 눈 왼쪽 끝
            shape[45],     # 오른쪽 눈 오른쪽 끝
            shape[48],     # 왼쪽 입술
            shape[54]      # 오른쪽 입술
        ], dtype="double")

        dist_coeffs = np.zeros((4,1)) # 왜곡 계수 없음
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        # 각도 변환
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles  # 고개 각도

        cv2.putText(frame, f"Pitch: {int(pitch)} Yaw: {int(yaw)} Roll: {int(roll)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Blink & Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
