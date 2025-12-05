# Turn detection states into commands (from detector.py output)

"""
command_mapper.py
------------------
Maps detection states from detector.py into high-level commands for Unity.

Expected detector.py output format (example):
{
    "left_eye": "closed" or "open",
    "right_eye": "closed" or "open",
    "mouth": "open" or "closed"
}

Output command format (example):
{
    "steer": -1 | 0 | 1,   # -1=left, 1=right, 0=straight
    "accelerate": True/False,
    "brake": True/False
}
"""

class CommandMapper:
    def __init__(self):
        pass

    def map_detection_to_command(self, detection_state: dict) -> dict:
        """
        Converts detector.py output → driving commands.
        """

        left_eye = detection_state.get("left_eye", "open")
        right_eye = detection_state.get("right_eye", "open")
        mouth = detection_state.get("mouth", "closed")

        # Determine steering
        if left_eye == "closed" and right_eye == "open":
            steer = -1  # turn left
        elif right_eye == "closed" and left_eye == "open":
            steer = 1   # turn right
        else:
            steer = 0   # no steering change

        # Determine acceleration & brake
        accelerate = mouth == "open"
        brake = mouth == "closed"

        return {
            "steer": steer,
            "accelerate": accelerate,
            "brake": brake,
        }

# Example usage (for controller.py):
# mapper = CommandMapper()
# commands = mapper.map_detection_to_command(detector_output)
# unity_sender.send(commands)
