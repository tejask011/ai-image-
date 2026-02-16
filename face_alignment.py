import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def align_face(image, bbox):
    xmin, ymin, width, height = bbox

    # Crop face safely
    face = image[ymin:ymin+height, xmin:xmin+width]
    if face.size == 0:
        return None

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    h, w, _ = face.shape

    # eye landmarks
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]

    left = np.array([left_eye.x * w, left_eye.y * h])
    right = np.array([right_eye.x * w, right_eye.y * h])

    # angle between eyes
    dy = right[1] - left[1]
    dx = right[0] - left[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # rotate face
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(face, M, (w, h))

    # standard size for embedding model
    aligned = cv2.resize(aligned, (160, 160))

    return aligned
