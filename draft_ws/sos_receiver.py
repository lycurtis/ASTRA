# import socket
# import time

# UDP_IP = "0.0.0.0"  # Listen on all network interfaces
# UDP_PORT = 5005  # Same port as sender

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((UDP_IP, UDP_PORT))

# print(f"Listening for YOLOv8 detections on port {UDP_PORT}...")

# last_play_time = 0  # To avoid rapid sound playing
# sound_file = "alert.mp3"  # Replace with your sound file (.mp3 or .wav)

# while True:
#     data, addr = sock.recvfrom(1024)  # Receive UDP packet
#     message = data.decode()
#     print(f"Message from Jetson: {message}")

#     # If a person is detected, play the alert sound
#     if message == "DETECTED":
#         current_time = time.time()
#         if current_time - last_play_time > 2:  # Prevents continuous sound spam
#             print("Playing alert sound!")
#             playsound(sound_file)
#             last_play_time = current_time  # Update last play time

#!/usr/bin/env python3

import socket
import time
from datetime import datetime
import csv
import os
import winsound
import logging

# Configuration
UDP_IP = "0.0.0.0"  # Listen on all network interfaces
UDP_PORT = 5005
BEEP_DURATION = 1000  # milliseconds
BEEP_FREQ = 1000  # Hz
MIN_SOUND_INTERVAL = 2  # Minimum seconds between alert sounds
LOG_FILE = "detections.csv"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DetectionReceiver:
    def __init__(self):
        self.sock = None
        self.last_play_time = 0
        self.last_gps = None
        self.setup_socket()
        self.setup_csv()

    def setup_socket(self):
        """Initialize UDP socket with error handling"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((UDP_IP, UDP_PORT))
            logger.info(f"Listening for detections on port {UDP_PORT}...")
        except socket.error as e:
            logger.error(f"Socket error: {e}")
            raise

    def setup_csv(self):
        """Initialize CSV file for logging detections"""
        csv_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['Timestamp', 'Status', 'Latitude', 'Longitude', 'Altitude'])

    def play_alert(self):
        """Play alert beep with rate limiting"""
        current_time = time.time()
        if current_time - self.last_play_time > MIN_SOUND_INTERVAL:
            try:
                winsound.Beep(BEEP_FREQ, BEEP_DURATION)
                self.last_play_time = current_time
                #logger.info("Alert sound played")
            except Exception as e:
                logger.error(f"Error playing sound: {e}")

    def parse_message(self, message):
        """Parse incoming message and extract status and GPS data"""
        parts = message.split(',')
        status = parts[0]
        
        if status == "DETECTED" and len(parts) >= 4:
            try:
                lat, lon, alt = map(float, parts[1:4])
                return status, (lat, lon, alt)
            except ValueError:
                logger.warning("Invalid GPS data format")
                return status, None
        
        return status, None

    def log_detection(self, status, gps_data):
        """Log detection to CSV file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if gps_data:
                writer.writerow([timestamp, status, *gps_data])
            else:
                writer.writerow([timestamp, status, 'N/A', 'N/A', 'N/A'])

    def display_info(self, status, gps_data):
        """Display formatted detection information"""
        if status == "DETECTED":
            if gps_data:
                lat, lon, alt = gps_data
                logger.info("PERSON DETECTED at coordinates:")
                logger.info(f"  Latitude:  {lat:.6f}")
                logger.info(f"  Longitude: {lon:.6f}")
                logger.info(f"  Altitude:  {alt:.2f}m")
            else:
                logger.info("PERSON DETECTED - GPS data unavailable")
        else:
            logger.info("Searching for persons...")

    def run(self):
        """Main reception loop"""
        try:
            while True:
                data, addr = self.sock.recvfrom(1024)
                message = data.decode().strip()
                
                status, gps_data = self.parse_message(message)
                self.display_info(status, gps_data)
                self.log_detection(status, gps_data)

                if status == "DETECTED":
                    self.play_alert()

        except KeyboardInterrupt:
            logger.info("Shutting down receiver...")
            self.sock.close()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.sock.close()

if __name__ == "__main__":
    try:
        receiver = DetectionReceiver()
        receiver.run()
    except Exception as e:
        logger.error(f"Failed to start receiver: {e}")
