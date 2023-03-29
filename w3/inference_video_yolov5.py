import cv2
import torch
import os

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

print(os.path.abspath('vdo.mp4'))

# Open video file
cap = cv2.VideoCapture('vdo.mp4')

# Check if video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

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

    # Write frame to output video
    out.write(cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR))

    # Display the frame
    cv2.imshow('Frame', results.ims[0])

    # Check if user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video file, video writer and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
