# eye aspect ratio
# mouth aspect ratio

import math
import numpy as np

# Indices for landmarks from MediaPipe Face Mesh
# For eyes we use 6 points each:
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144] # clockwise ish ig haha idk
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]# For mouth we use inner upper/lower and corners:
MOUTH_TOP = 13     # upper inner lip
MOUTH_BOTTOM = 14  # lower inner lip
MOUTH_LEFT = 61    # left corner
MOUTH_RIGHT = 291  # right corner



def _dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def landmark_to_point(landmark, w, h):
    return np.array([int(landmark.x * w), int(landmark.y * h)], dtype=np.float32)

def eye_aspect_ratio(landmarks, indices, w, h):
    """
    Compute EAR using six landmarks similar to the classic formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    pts = [landmark_to_point(landmarks[i], w, h) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    # vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    # horizontal distance
    hdist = np.linalg.norm(p1 - p4)
    if hdist == 0:
        return 0.0
    ear = (v1 + v2) / (2.0 * hdist)
    return ear

def get_left_ear(landmarks, w, h):
    return eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)

def get_right_ear(landmarks, w, h):
    return eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w ,h )

def mouth_aspect_ratio(landmarks):
    """
    MAR = vertical distance (top-bottom inner lip) / mouth width (corners)
    """
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = _dist(top, bottom)
    horizontal = _dist(left, right)
    if horizontal == 0:
        return 0.0
    mar = vertical / horizontal
    return mar
