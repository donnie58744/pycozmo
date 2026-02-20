import sys
import os

"""REMOVE IN PRODUCTION WAITING FOR TTS PR TO BE MERGED"""
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.normpath(os.path.join(current_dir, "../"))  # parent of pycozmo
sys.path.insert(0, lib_path)  # insert at front to take priority

print("Looking in:", lib_path)
print("pycozmo exists:", os.path.exists(os.path.join(lib_path, "pycozmo")))

import pycozmo


if __name__ == "__main__":
    robot = pycozmo.Client()
    robot.start()
    robot.connect()
    robot.wait_for_robot()
    robot.set_volume(50000)
    robot.say_text(txt="Hello World, My name is Cozmo!")
    robot.wait_for(pycozmo.event.EvtAudioCompleted)