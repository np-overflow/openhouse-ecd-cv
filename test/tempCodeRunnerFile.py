#for testing dectector.py (duh)
from detection.detector import ExpressionDetector

detector = ExpressionDetector()

while True:
    state = detector.get_detection_state()
    print(state)
