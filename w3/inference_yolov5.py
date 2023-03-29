import cv2
import torch
import os
import json

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

print(os.path.abspath('vdo.mp4'))

# Open video file
cap = cv2.VideoCapture('vdo.mp4')

# Check if video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Create empty list for bounding boxes
bbxs = []

# Loop through each frame of the video
while cap.isOpened():
    # Read frame from video
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        print("Error reading frame")
        break

    # Convert frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference with YOLOv5
    results = model([frame])
    car_results = results.pred[0][results.pred[0][:, 5] == 2]
    results.render()

    # Add bounding boxes to list
    bbxs.append(results.pandas().xyxy[0].to_dict())

    # Display the frame
    cv2.imshow('Frame', results.ims[0])

    # Check if user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save bounding boxes to JSON file
with open('bbxs.json', 'w') as f:
    json.dump(bbxs, f)

# Release video file and close windows
cap.release()
cv2.destroyAllWindows()
