import socket

UDP_PORT = 5005  # Must match the sender's port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))

print(f"Listening for GPS data on port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(1024)
    gps_data = data.decode().split(",")
    lat, lon, alt = map(float, gps_data)
    print(f"[Drone Location] Latitude: {lat}, Longitude: {lon}, Altitude: {alt}m")
