# detector.py
from detection.face_landmarks import FaceLandmarkDetector
from detection.ratios import get_left_ear, get_right_ear, mouth_aspect_ratio

class ExpressionDetector:
    def __init__(self):
        self.landmark_detector = FaceLandmarkDetector()
        self.landmark_detector.start()  # keep webcam open continuously

        # Hysteresis thresholds
        self.LEFT_EYE_CLOSE = 0.18
        self.LEFT_EYE_OPEN  = 0.23
        self.RIGHT_EYE_CLOSE = 0.18
        self.RIGHT_EYE_OPEN  = 0.23

        self.MAR_THRESHOLD = 0.3

        # Buffers for smoothing
        self.left_ear_buffer = []
        self.right_ear_buffer = []
        self.buffer_size = 4 # number of frames to average

        # Previous states for hysteresis
        self.prev_left_eye = "open"
        self.prev_right_eye = "open"

    def get_detection_state(self, draw_frame=False):
        success, frame, landmarks = self.landmark_detector.get_landmarks(draw=draw_frame)
        if not success or landmarks is None:
            return {"left_eye": "open", "right_eye": "open", "mouth": "closed"}, frame

        # Compute EAR/MAR
        left_ear = get_left_ear(landmarks)
        right_ear = get_right_ear(landmarks)
        mar = mouth_aspect_ratio(landmarks)

        # Update buffers
        self.left_ear_buffer.append(left_ear)
        self.right_ear_buffer.append(right_ear)
        if len(self.left_ear_buffer) > self.buffer_size:
            self.left_ear_buffer.pop(0)
            self.right_ear_buffer.pop(0)

        # Smoothed EARs
        left_ear_avg = sum(self.left_ear_buffer) / len(self.left_ear_buffer)
        right_ear_avg = sum(self.right_ear_buffer) / len(self.right_ear_buffer)

        # Mouth open/closed
        mouth_state = "open" if mar > self.MAR_THRESHOLD else "closed"

        # Hysteresis logic for eyes
        if self.prev_left_eye == "open":
            left_eye_state = "closed" if left_ear_avg < self.LEFT_EYE_CLOSE else "open"
        else:  # previously closed
            left_eye_state = "open" if left_ear_avg > self.LEFT_EYE_OPEN else "closed"

        if self.prev_right_eye == "open":
            right_eye_state = "closed" if right_ear_avg < self.RIGHT_EYE_CLOSE else "open"
        else:
            right_eye_state = "open" if right_ear_avg > self.RIGHT_EYE_OPEN else "closed"

        # Save previous states
        self.prev_left_eye = left_eye_state
        self.prev_right_eye = right_eye_state

        detection_state = {
            "left_eye": left_eye_state,
            "right_eye": right_eye_state,
            "mouth": mouth_state
        }

        return detection_state, frame

# Quick test
if __name__ == "__main__":
    detector = ExpressionDetector()
    print("Press q to quit.")
    while True:
        state, _ = detector.get_detection_state()
        print(state)
