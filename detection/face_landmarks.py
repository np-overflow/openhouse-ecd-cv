# Set up webcam
# Run MediaPipe Face Mesh to detect face landmarks
# Return landmarks for further processing

# detection/face_landmarks.py
import cv2
import mediapipe as mp

class FaceLandmarkDetector:
    def __init__(self, cam_index=0, max_faces=1, static_image_mode=False, refine_landmarks=False):
        self.cam_index = cam_index
        self.cap = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.cam_index)

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def _normalized_to_pixel(self, lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def get_frame_landmarks(self, draw=False):
        """
        Read one frame from webcam, return:
          - success (bool)
          - frame (BGR image)
          - landmarks (list of (x,y) tuples) or None if no face
        """
        if self.cap is None:
            self.start()

        ret, frame = self.cap.read()
        if not ret:
            return False, None, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            if draw:
                cv2.imshow("FaceMesh", frame)
                cv2.waitKey(1)
            return True, frame, None

        # Only use first detected face
        face_landmarks = results.multi_face_landmarks[0]
        pts = []
        for lm in face_landmarks.landmark:
            pts.append(self._normalized_to_pixel(lm, w, h))

        if draw:
            # draw landmarks (small dots)
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            cv2.imshow("FaceMesh", frame)
            cv2.waitKey(1)

        return True, frame, pts

if __name__ == "__main__":
    # quick manual test
    detector = FaceLandmarkDetector()
    detector.start()
    print("Press 'q' in the image window to quit.")
    while True:
        success, frame, pts = detector.get_frame_landmarks(draw=True)
        if not success:
            break
        # quit if user presses 'q' in the OpenCV window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    detector.stop()
