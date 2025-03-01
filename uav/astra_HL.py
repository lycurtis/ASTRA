"""
Code Description: Headless human detection and log writing + optimization
"""
#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import threading
from flask import Flask, Response
import os
import queue

# Disable display for OpenCV GUI
os.environ["DISPLAY"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Flask App
app = Flask(__name__)

# Get YOLOv8n model blob file path
nnPath = str((Path(__file__).parent / Path('../models/yolov8n_coco_640x352.blob')).resolve().absolute())
if not Path(nnPath).exists():
    raise FileNotFoundError('Required file/s not found, please install the required models.')

# Label text for YOLOv8
labelMap = ["person"]
syncNN = True

# Log file for detections
LOG_FILE = "/tmp/human_detect_log.txt"

def write_to_log(message):
    """Write detection result to log file"""
    with open(LOG_FILE, "w") as f:
        f.write(message + "\n")
        
        
# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Camera properties
camRgb.setPreviewSize(640, 352)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)  # Reduced FPS for efficiency

# YOLOv8 network configuration
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([])
detectionNetwork.setAnchorMasks({})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Global variables
frame_queue = queue.Queue(maxsize=1)
lock = threading.Lock()
detections = []
last_logged_message = None

# Function to normalize bounding box coordinates
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Optimized frame display
def displayFrame(frame, detections_list):
    global last_logged_message
    display_frame = frame.copy()
    person_detected = False
    
    for detection in detections_list:
        if detection.label == 0:  # Person class in COCO
            person_detected = True
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            confidence_text = f"Person: {detection.confidence * 100:.1f}%"
            cv2.putText(display_frame, confidence_text, (bbox[0], bbox[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Store the latest frame in the queue
    if not frame_queue.full():
        frame_queue.put(display_frame)
        
    # Write detection status to log file only if it changed
    log_message = "DETECTED" if person_detected else "NOT_DETECTED"
    if log_message != last_logged_message:
        write_to_log(log_message)
        last_logged_message = log_message

# Flask Route to Stream Video
def generate_frames():
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask Server in a Separate Thread
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

def main():
    # Start Flask Thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start DepthAI Pipeline
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

        frame = None  # Initialize frame variable
        
        print('Starting detection...')
        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()
            
            if inRgb is not None:
                frame = inRgb.getCvFrame()
            
            if inDet is not None:
                detections = inDet.detections
                if frame is not None:
                    displayFrame(frame, detections)
            
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    main()
