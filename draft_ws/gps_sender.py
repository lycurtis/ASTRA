import socket
from pymavlink import mavutil

# Setup UDP connection
GROUND_STATION_IP = "192.168.1.121"  # Replace with your receiever (laptop's) IP
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Connect to Pixhawk
master = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)
master.wait_heartbeat()
print("Connected to Pixhawk!")

# Function to get GPS coordinates
def get_gps():
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        latitude = msg.lat / 1e7
        longitude = msg.lon / 1e7
        altitude = msg.alt / 1000.0
        return latitude, longitude, altitude
    return None

# Main loop: Read GPS and send over UDP
while True:
    gps_data = get_gps()
    if gps_data:
        message = f"{gps_data[0]},{gps_data[1]},{gps_data[2]}"
        sock.sendto(message.encode(), (GROUND_STATION_IP, UDP_PORT))
        print(f"Sent: {message}")
