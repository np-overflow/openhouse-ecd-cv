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
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
        