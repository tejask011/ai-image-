from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os

from face_detection import detect_faces_from_array
from face_alignment import align_face

app = Flask(__name__)

# ensure static folder exists
os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # STEP 1: detect faces
    faces = detect_faces_from_array(image)

    results = []

    # STEP 2: align each face
    for i, f in enumerate(faces):
        bbox = (f["xmin"], f["ymin"], f["width"], f["height"])

        aligned = align_face(image, bbox)

        if aligned is None:
            continue

        # save aligned face for testing
        path = f"static/aligned_{i}.jpg"
        cv2.imwrite(path, aligned)

        results.append({
            "face_id": i,
            "bbox": f,
            "aligned_face": path
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
