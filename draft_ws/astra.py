# import subprocess

# # Commands to open scripts in new terminal tabs
# commands = [
#     "gnome-terminal --tab -- bash -c 'python3 /home/uav/depthai-python/examples/Yolo/human_detect.py; exec bash'",
#     "gnome-terminal --tab -- bash -c 'python3 /home/uav/astra/src/sos_transmitter.py; exec bash'"
# ]

# # Launch scripts in separate terminal tabs
# for command in commands:
#     subprocess.Popen(command, shell=True)

#Without terminal (Runs in background)
import subprocess

scripts = [
    "/home/uav/depthai-python/examples/Yolo/human_detect.py",
    "/home/uav/astra/src/sos_transmitter.py"
]

processes = [subprocess.Popen(["python3", script]) for script in scripts]
