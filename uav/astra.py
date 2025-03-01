"""
Code Description: This is a launch script that will run all necessary UAV scripts
"""

"""
#Non-headless
import subprocess

# Commands to open scripts in new terminal tabs
commands = [
    "gnome-terminal --tab -- bash -c 'python3 /home/uav/depthai-python/examples/Yolo/human_detect.py; exec bash'",
    "gnome-terminal --tab -- bash -c 'python3 /home/uav/astra/src/sos_transmitter.py; exec bash'"
]

# Launch scripts in separate terminal tabs
for command in commands:
    subprocess.Popen(command, shell=True)
"""

#Completely heaadless
import subprocess

# Commands to launch scripts in the background
commands = [
    "python3 /home/uav/depthai-python/examples/Yolo/human_detect.py",
    "python3 /home/uav/astra/src/sos_transmitter.py"
]

# Launch scripts as independent background processes
processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]

# Keep the main script running
for process in processes:
    process.wait()
