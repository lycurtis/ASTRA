from ultralytics import YOLO
import cvzone
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLO model
model = YOLO('yolov10n.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True
)

# Video input - replace 'path/to/your/video.mp4' with your video file path
video_path = 'D:/Dworkspace/traffic.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer (optional - if you want to save the output video)
output_path = 'output.mp4'
output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, image = cap.read()
    if not ret:
        break
        
    # YOLO detection
    results = model(image)
    
    # Prepare detections for DeepSORT
    detections_for_tracker = []
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            conf = float(box.conf[0].numpy())
            cls = int(box.cls[0])
            class_name = results[0].names[cls]
            
            # Format for DeepSORT: [x1,y1,w,h] (without confidence)
            w = x2 - x1
            h = y2 - y1
            detection = [x1, y1, w, h]
            detections_for_tracker.append((detection, conf, class_name))
    
    # Update tracker
    tracks = tracker.update_tracks(detections_for_tracker, frame=image)
    
    # Draw tracked objects
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ID and class name
        label = f"ID: {track_id}"
        cvzone.putTextRect(image, label, [x1 + 8, y1 - 12], 
                         thickness=2, scale=1.5)

    # Write frame to output video (optional)
    output.write(image)
    
    # Display frame
    cv2.imshow('frame', image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
output.release()  # Release the output video writer
cv2.destroyAllWindows()
