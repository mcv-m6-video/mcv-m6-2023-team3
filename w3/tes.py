import cv2
import os

# Open the video file
video = cv2.VideoCapture('vdo.avi')
print(os.listdir())
# Get the current working directory
current_dir = os.getcwd()

# Print the current working directory
print("Current working directory:", current_dir)

# Check if the video was opened successfully
if not video.isOpened():
    print("Error opening video file")

# Read and save each frame of the video
frame_count = 0
while True:
    ret, frame = video.read()

    # Check if there are no more frames to read
    if not ret:
        break

    # Save the frame to a file
    filename = f"frame_{frame_count}.jpg"
    cv2.imwrite(filename, frame)

    # Increment the frame count
    frame_count += 1

# Release the video file and close all windows
video.release()