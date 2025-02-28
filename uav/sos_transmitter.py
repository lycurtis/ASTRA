"""
Code Description: Reads from a log file and processes the input data (DETECTED OR NOT_DETECTED) + transmit output to ground station 
"""
#!/usr/bin/env python3

#When the drone identifies a person it will send out a signal to the ground station the cooordinates of the drone

import socket
import time

#UDP Configuration (Send to PC/Ground Station)
UDP_IP = "192.168.1.106" #Replace with your PC's IP address
UDP_PORT = 5005 #Port for sending detection message
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP Socket

#File where 'human_detect.py' logs 'DETECTED' 
LOG_FILE = "/tmp/human_detect_log.txt" #Make sure 'human_detect.py' writes to this file

print("Monitoring human_detect.py for detection events...")

def send_udp_message(message):
  #Send detection status over WiFi (UDP)
  sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
last_sent_message = None #Prevents duplicate messages from being sent

while True:
  try:
    #Read latest detection result from the log file
    with open(LOG_FILE, "r") as f:
      lines = f.readlines()
      if lines:
        last_line = lines[-1].strip() #Get the last logged detection result

        #Send only if the message has changed
        if last_line in ["DETECTED", "NOT_DETECTED"] and last_line != last_sent_message:
          send_udp_message(last_line)
          print(f"Sent to PC: {last_line}")
          last_sent_message = last_line #Update last sent message
    time.sleep(1) 

except KeyboardInterrupt:
  print("Stopping object detection transmitter...")
  sock.close()
  break
except Exception as e:
  print(f"Error: {e}")
  time.sleep(1)
