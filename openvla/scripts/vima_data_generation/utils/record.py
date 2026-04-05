import cv2
from pyvirtualdisplay.smartdisplay import SmartDisplay
import numpy as np


class Recorder:
    def __init__(self, width, height, fps):
        self.video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        self.display = SmartDisplay(visible=False, size=(width, height))

    def __enter__(self):
        self.display.start()
        return self

    def record_once(self):
        data = cv2.cvtColor(np.array(self.display.waitgrab()), cv2.COLOR_RGB2BGR)
        self.video_writer.write(data)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.display.stop()
        self.video_writer.release()
        print("Exiting from Recorder...")


class VirtualDisplay():
    def __init__(self, width, height):
        self.display = SmartDisplay(visible=False, size=(width, height))

    def __enter__(self):
        self.display.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.display.stop()
        print("Exiting from VirtualDisplay...")
