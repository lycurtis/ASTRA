import subprocess

# Commands to open scripts in new terminal tabs
commands = [
    "gnome-terminal --tab -- bash -c 'python3 /home/uav/depthai-python/examples/Yolo/human_detect.py; exec bash'",
    "gnome-terminal --tab -- bash -c 'python3 /home/uav/astra/src/sos_transmitter.py; exec bash'"
]

# Launch scripts in separate terminal tabs
for command in commands:
    subprocess.Popen(command, shell=True)
