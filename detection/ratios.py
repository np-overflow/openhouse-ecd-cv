# eye aspect ratio
# mouth aspect ratio

import math

# Indices for landmarks from MediaPipe Face Mesh
# For eyes we use 6 points each:
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144] # clockwise ish ig haha idk
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# For mouth we use inner upper/lower and corners:
MOUTH_TOP = 13     # upper inner lip
MOUTH_BOTTOM = 14  # lower inner lip
MOUTH_LEFT = 61    # left corner
MOUTH_RIGHT = 291  # right corner

def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(landmarks, eye_indices):
    """
    landmarks: list of (x,y) tuples length 468
    eye_indices: 6 indices: [p1, p2, p3, p4, p5, p6]
    returns EAR (float)
    Formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = _dist(p2, p6)
    vertical2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def get_left_ear(landmarks):
    return eye_aspect_ratio(landmarks, LEFT_EYE_IDX)

def get_right_ear(landmarks):
    return eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)

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
