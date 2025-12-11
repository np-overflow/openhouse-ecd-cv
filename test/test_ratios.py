# print EAR and MAR values while testing

from detection.face_landmarks import FaceLandmarkDetector
from detection.ratios import get_left_ear, get_right_ear, mouth_aspect_ratio
import cv2

detector = FaceLandmarkDetector()
detector.start()

print("Press 'q' to quit.")

while True:
    success, frame, landmarks = detector.get_landmarks(draw=True)
    if not success or landmarks is None:
        continue

    left_ear = get_left_ear(landmarks)
    right_ear = get_right_ear(landmarks)
    mar = mouth_aspect_ratio(landmarks)

    print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, MAR: {mar:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.stop()
