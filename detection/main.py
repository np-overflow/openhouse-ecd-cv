from detection.ratios import get_left_ear, get_right_ear, mouth_aspect_ratio
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
# detection/face_landmarks.py

    # -------------------------
# Configuration (tweak me)
# -------------------------
# Eye Aspect Ratio threshold below which eye is considered "closed"
EAR_THRESHOLD = 0.1
MAR_TRESHOLD = 0.5
# Number of consecutive frames with EAR < threshold to trigger alarm
CONSEC_FRAMES = 20
SMOOTH_WINDOW = 6           # Moving average window for EAR to reduce flicker
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144] # clockwise ish ig haha idk
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]# For mouth we use inner upper/lower and corners:
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    left_ear_history = deque(maxlen=SMOOTH_WINDOW)
    right_ear_history = deque(maxlen=SMOOTH_WINDOW)
    mar_history = deque(maxlen=SMOOTH_WINDOW)
    left_counter = 0
    right_counter = 0
    mar_counter = 0
    left_eye_closed = False
    right_eye_closed = False
    mar_closed = False
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Can't read from camera.")
                break

            h, w = frame.shape[:2]
            # flip horizontally so it feels like a mirror (optional)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                left_ear = get_left_ear(face_landmarks, w, h)
                right_ear = get_right_ear(
                    face_landmarks, w, h)
                mar = mouth_aspect_ratio(face_landmarks)
                mar_history.append(mar)
                mar_smooth = float(np.mean(mar_history))
            
                left_ear_history.append(left_ear)
                right_ear_history.append(right_ear)
                left_smooth = float(np.mean(left_ear_history))
                right_smooth = float(np.mean(right_ear_history))
                # Draw EAR on frame
                cv2.putText(frame, f"LEFT EAR: {left_smooth:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if left_smooth > EAR_THRESHOLD else (0, 0, 255), 2)
                
                cv2.putText(frame, f"RIGHT EAR: {right_smooth:.3f}", (350, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if right_smooth > EAR_THRESHOLD else (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mar_smooth:.3f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # Check drowsiness
                if left_smooth < EAR_THRESHOLD:
                    left_counter += 1
                else:
                    left_counter = 0
                    left_eye_closed = False

                if right_smooth < EAR_THRESHOLD:
                    right_counter += 1
                else:
                    right_counter = 0
                    right_eye_closed = False

                if mar_smooth < MAR_TRESHOLD:
                    mar_counter += 1
                else:
                    mar_counter = 0
                    mar_closed = False


                # Trigger detection if eyes closed for too long, ensures no unplanning turns due to twitching
                if left_counter >= CONSEC_FRAMES and not left_eye_closed:
                    left_eye_closed = True

                if right_counter >= CONSEC_FRAMES and not right_eye_closed:
                    right_eye_closed = True

                if mar_counter >= CONSEC_FRAMES and not mar_closed:
                    mar_closed = True
                # Visual alert
                if left_eye_closed:
                    cv2.putText(frame, "LEFT EYE CLOSED", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if right_eye_closed:
                    cv2.putText(frame, "RIGHT EYE CLOSED", (350, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if mar_closed:
                    cv2.putText(frame, "MOUTH CLOSED", (350, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
          
            else:
                # no face detected: reset counters and show message
                left_ear_history.clear()
                right_ear_history.clear()
                left_counter = 0
                right_counter = 0
                mar_counter = 0
                mar_closed = False
                left_eye_closed = False
                right_eye_closed = False
                cv2.putText(frame, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # quick manual test
    main()
