import socket
import time
import winsound

# UDP Setup
UDP_IP = "0.0.0.0"  # Listen on all network interfaces
UDP_PORT = 5005     # Same port as sender

# Sound settings
BEEP_FREQ = 750    # Frequency in Hz
BEEP_DURATION = 100 # Duration in milliseconds

# Create and bind socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for detections on port {UDP_PORT}...")

last_play_time = 0  # To avoid rapid sound playing

while True:
    try:
        data, addr = sock.recvfrom(1024)  # Receive UDP packet
        message = data.decode().strip()
        
        # Parse the message
        parts = message.split(',')
        status = parts[0]

        # Handle detection
        if status == "DETECTED":
            # If GPS data is included, display it
            if len(parts) >= 4:
                lat, lon, alt = map(float, parts[1:4])
                print(f"\nPERSON DETECTED at coordinates:")
                print(f"Latitude:  {lat:.6f}")
                print(f"Longitude: {lon:.6f}")
                print(f"Altitude:  {alt:.2f}m")
            else:
                print("\nPERSON DETECTED!")

            # Play alert sound with rate limiting
            current_time = time.time()
            if current_time - last_play_time > 0.5:  # Wait at least 1 second between beeps
                winsound.Beep(BEEP_FREQ, BEEP_DURATION)
                last_play_time = current_time
        
        elif status == "NOT_DETECTED":
            print("\nSearching for persons...")

    except KeyboardInterrupt:
        print("\nStopping receiver...")
        sock.close()
        break
    except Exception as e:
        print(f"\nError: {e}")
        time.sleep(1)
