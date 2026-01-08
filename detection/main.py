from detection.ratios import get_left_ear, get_right_ear, mouth_aspect_ratio
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import asyncio
import json
import websockets
from websockets.asyncio.server import serve
import os

# detection/face_landmarks.py

    # -------------------------
# Configuration (tweak me)
# -------------------------
# Eye Aspect Ratio threshold below which eye is considered "closed"
EAR_THRESHOLD_SMALL = 0.11
EAR_THRESHOLD_LARGE = 0.16
MAR_CLOSED_THRESHOLD = 0.5
MAR_HALFOPEN_THRESHOLD = 1
# Number of consecutive frames with EAR < threshold to trigger alarm
CONSEC_FRAMES = 5
SMOOTH_WINDOW = 6           # Moving average window for EAR to reduce flicker
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144] # clockwise ish ig haha idk
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]# For mouth we use inner upper/lower and corners:
PORT = 8765
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
count = 0
with open("photocounter.txt","r") as counter:
    count = counter.readline().strip()
    count = int(count)
current_frame = None
frame_lock = asyncio.Lock()
data = {
    "move": 0,
    "steer_s": 0,
    "steer_l": 0
}
def capture(frame):
    global count
    count += 1
    outputdir = "saved_images"
    savepath = os.path.join(outputdir,f"IMG_{count}.png")
    cv2.imwrite(savepath,frame)

def scale(img, zoom_factor=0.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

def zoom_center_crop(image, scale_factor=1.5, height_offset=50):
    """Zooms into the center of an image by cropping and resizing."""
    height, width, channels = image.shape
    
    # Calculate the new dimensions for the cropped area
    # Scale factor > 1 means zoom in
    new_height = int(height / scale_factor)
    new_width = int(width / scale_factor)
    
    # Calculate the center of the image
    center_y, center_x = height // 2, width // 2
    
    # Calculate the top-left and bottom-right coordinates for the crop
    # Ensure coordinates are within image boundaries
    min_x = max(0, center_x - new_width // 2)
    max_x = min(width, center_x + new_width // 2)
    min_y = max(0, center_y - new_height // 2) - height_offset
    max_y = min(height, center_y + new_height // 2) - height_offset

    # Crop the image (ROI selection)
    cropped = image[min_y:max_y, min_x:max_x]
    
    # Resize the cropped region back to the original dimensions
    # This enlarges the pixels of the cropped area to fill the frame
    zoomed_image = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return zoomed_image

async def handler(websocket):
    try:
        while True:
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.05)
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                print("Received:", msg)

                if msg == "capture":
                    capture(current_frame)
            except asyncio.TimeoutError:
                pass 
    except websockets.exceptions.ConnectionClosed:
        print("CLIENT DISCONNECTED")
async def servermain():
    async with serve(handler, "localhost", PORT):
        print(f"WebSocket running on ws://localhost:{PORT}")
        await asyncio.Future()
async def cv_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    left_ear_history = deque(maxlen=SMOOTH_WINDOW)
    right_ear_history = deque(maxlen=SMOOTH_WINDOW)
    mar_history = deque(maxlen=SMOOTH_WINDOW)
    left_counter_small = 0
    right_counter_small = 0
    left_counter_large = 0
    right_counter_large = 0
    mar_closed_counter = 0
    mar_halfopen_counter = 0
    left_eye_closed_s = False
    left_eye_closed_l = False
    right_eye_closed_s = False
    right_eye_closed_l = False
    mar_state = "stopped"
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

            # frame resizing
            frame = scale(frame, 0.5)
            frame = zoom_center_crop(frame, 2.2, 40)

            async with frame_lock:
                current_frame = frame.copy()
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if left_smooth > EAR_THRESHOLD_SMALL else (0, 0, 255), 2)
                
                cv2.putText(frame, f"RIGHT EAR: {right_smooth:.3f}", (350, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if right_smooth > EAR_THRESHOLD_SMALL else (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mar_smooth:.3f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, str(data["move"]), (60, 250),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, str(data["steer_s"]), (90, 300),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, str(data["steer_l"]), (90, 350),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Check drowsiness
                if left_smooth < EAR_THRESHOLD_SMALL:
                    left_counter_small += 1
                else:
                    left_counter_small = 0
                    left_eye_closed_s = False

                if left_smooth < EAR_THRESHOLD_LARGE:
                    left_counter_large += 1
                else:
                    left_counter_large = 0
                    left_eye_closed_l = False

                if right_smooth < EAR_THRESHOLD_SMALL:
                    right_counter_small += 1
                else:
                    right_counter_small = 0
                    right_eye_closed_s = False

                if right_smooth < EAR_THRESHOLD_LARGE:
                    right_counter_large += 1
                else:
                    right_counter_large = 0
                    right_eye_closed_l = False

                if mar_smooth < MAR_CLOSED_THRESHOLD:
                    mar_closed_counter += 1
                    mar_halfopen_counter = 0
                elif mar_smooth < MAR_HALFOPEN_THRESHOLD:
                    mar_halfopen_counter += 1
                    mar_closed_counter = 0
                else:
                    mar_closed_counter = 0
                    mar_halfopen_counter = 0
                    mar_state = "open"


                # small and QUICK 
                # Trigger detection if eyes closed for too long, ensures no unplanning turns due to twitching
                if left_counter_small >= CONSEC_FRAMES and not left_eye_closed_s:
                    left_eye_closed_s = True

                if left_counter_large >= CONSEC_FRAMES and not left_eye_closed_l:
                    left_eye_closed_l = True

                if right_counter_small >= CONSEC_FRAMES and not right_eye_closed_s:
                    right_eye_closed_s = True
                
                if right_counter_large >= CONSEC_FRAMES and not right_eye_closed_l:
                    right_eye_closed_l = True

                if mar_closed_counter >= CONSEC_FRAMES and not mar_state == "closed":
                    mar_state = "closed"
                
                if mar_halfopen_counter >= CONSEC_FRAMES and not mar_state == "halfopen":
                    mar_state = "halfopen"
                
                # -- LARGE MODE STEERING
                if not left_eye_closed_l and not right_eye_closed_l:
                    data["steer_l"] = 0
                # Visual alert
                if left_eye_closed_l:
                    if data["steer_l"] > -0.2:
                        data["steer_l"] -= 0.05
                    elif data["steer_l"] > -0.4:
                         data["steer_l"] -= 0.03
                    elif data["steer_l"] > -1:
                        data["steer_l"] -= 0.01
                    else:
                        data["steer_l"] = -1
                    cv2.putText(frame, "LEFT EYE CLOSED", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                if right_eye_closed_l:
                    if data["steer_l"] < 0.2:
                        data["steer_l"] += 0.05
                    elif data["steer_l"] < 0.4:
                         data["steer_l"] += 0.03
                    elif data["steer_l"] < 1:
                        data["steer_l"] += 0.01
                    else:
                        data["steer_l"] = 1
                    cv2.putText(frame, "RIGHT EYE CLOSED", (350, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    
                # -- SMALL MODE STEERING --
                if not left_eye_closed_s and not right_eye_closed_s:
                    data["steer_s"] = 0
                # Visual alert
                if left_eye_closed_s:
                    if data["steer_s"] > -0.2:
                        data["steer_s"] -= 0.05
                    elif data["steer_s"] > -0.4:
                         data["steer_s"] -= 0.03
                    elif data["steer_s"] > -1:
                        data["steer_s"] -= 0.01
                    else:
                        data["steer_s"] = -1
                    cv2.putText(frame, "LEFT EYE CLOSED", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if right_eye_closed_s:
                    if data["steer_s"] < 0.2:
                        data["steer_s"] += 0.05
                    elif data["steer_s"] < 0.4:
                         data["steer_s"] += 0.03
                    elif data["steer_s"] < 1:
                        data["steer_s"] += 0.01
                    else:
                        data["steer_s"] = 1
                    cv2.putText(frame, "RIGHT EYE CLOSED", (350, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # DRIVING    
                
                if mar_state == "closed":
                    if data["move"] > 0:
                        data["move"] = 0
                    elif data["move"] > -1:
                        data["move"] -= 0.07
                    else:
                        data["move"] = -1
                    cv2.putText(frame, "MOUTH CLOSED", (350, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                elif mar_state == "halfopen":
                    if data["move"] < mar_smooth:
                        data["move"] += 0.05
                    elif data["move"] > mar_smooth:
                        data["move"] -= 0.05
                    cv2.putText(frame, "MOUTH HALF-OPEN", (350, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                elif mar_state == "stopped":
                    data["move"] = 0
                else:
                    if data["move"] < 0:
                        data["move"] = 0
                    elif data["move"] < 1:
                        data["move"] += 0.05
                    else:
                        data["move"] = 1
                        
          
            else:
                # no face detected: reset counters and show message
                left_ear_history.clear()
                right_ear_history.clear()
                left_counter = 0
                right_counter = 0
                mar_closed_counter = 0
                mar_halfopen_counter = 0
                mar_state = "stopped"
                left_eye_closed = False
                right_eye_closed = False
                cv2.putText(frame, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Detector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                with open("photocounter.txt","w") as counter:
                    counter.write(str(count))
                break

            await asyncio.sleep(0)

    cap.release()
    cv2.destroyAllWindows()

async def main():
    await asyncio.gather(
        servermain(),
        cv_loop()
    )



if __name__ == "__main__":
    asyncio.run(main())
