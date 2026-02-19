import sys
import os

# Get the directory of THIS file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to root, then into lib
lib_path = os.path.join(current_dir, "../..", "pycozmo/")

sys.path.append(lib_path)

import pycozmo


if __name__ == "__main__":
    robot = pycozmo.Client()
    robot.start()
    robot.connect()
    robot.wait_for_robot()
    robot.set_volume(3000)
    robot.say_text(txt="Hello World, My name is Cozmo!")
    robot.wait_for(pycozmo.event.EvtAudioCompleted)