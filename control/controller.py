# get data from detector.py
# map to commands using command_mapper.py
# send to unity using unity_sender.py

"""
controller.py
--------------
Main loop:
1. Get data from detector.py
2. Map to commands using command_mapper.py
3. Send to Unity using unity_sender.py
"""

from detector import Detector
from command_mapper import CommandMapper
from unity_sender import UnitySender
import time

class Controller:
    def __init__(self):
        self.detector = Detector()
        self.mapper = CommandMapper()
        self.sender = UnitySender()

    def run(self):
        print("Controller running. Press CTRL+C to stop.")
        while True:
            # 1. Read facial expression states
            detection_state = self.detector.get_state()

            # 2. Map detection → driving commands
            commands = self.mapper.map_detection_to_command(detection_state)

            # 3. Send commands to Unity
            self.sender.send(commands)

            # Loop timing (adjust as needed)
            time.sleep(0.05)

if __name__ == "__main__":
    controller = Controller()
    controller.run()
