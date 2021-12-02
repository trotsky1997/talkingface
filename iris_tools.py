from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite import IrisLandmark, iris_roi_from_face_landmarks
from fdlite.examples.iris_recoloring import recolor_iris
from PIL import Image

EXCITING_NEW_EYE_COLOR = (161, 52, 216)

detect_iris = IrisLandmark()


def iris(img, face_landmarks):
    # get ROI for both eyes
    eye_roi = iris_roi_from_face_landmarks(face_landmarks, img.shape[:2])
    left_eye_roi, right_eye_roi = eye_roi
    # detect iris landmarks for both eyes
    left_eye_results = detect_iris(img, left_eye_roi)
    right_eye_results = detect_iris(img, right_eye_roi, is_right_eye=True)
    print(left_eye_results.iris)
