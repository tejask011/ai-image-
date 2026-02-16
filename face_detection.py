import cv2
import mediapipe as mp

def detect_faces_from_array(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    results = face_detection.process(rgb)

    faces = []

    if results.detections:
        h, w, _ = image.shape

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            faces.append({
                "xmin": xmin,
                "ymin": ymin,
                "width": width,
                "height": height
            })

    return faces
