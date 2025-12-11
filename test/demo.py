# Test all here

import cv2
from detection.detector import ExpressionDetector
from detection.face_landmarks import draw_eye_mouth_state

detector = ExpressionDetector()

print("Press 'q' to quit.")
while True:
    state, frame = detector.get_detection_state(draw_frame=False)
    if frame is not None:
        frame = cv2.flip(frame, 1)  # mirror the frame

        # Draw landmarks mirrored
        _, _, landmarks = detector.landmark_detector.get_landmarks(draw=False)
        if landmarks:
            frame = draw_eye_mouth_state(frame, landmarks, state['left_eye'], state['right_eye'], state['mouth'], mirror=True)

        # Overlay text
        cv2.putText(frame, f"L Eye: {state['left_eye']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"R Eye: {state['right_eye']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Mouth: {state['mouth']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Face Demo", frame)

    print(state)  # prints detection states in console

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.landmark_detector.stop()
cv2.destroyAllWindows()
