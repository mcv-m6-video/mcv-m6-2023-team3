import glob
import os
import cv2
import matplotlib.pyplot as plt



def load_bb(path):
    """
    Loads the bounding boxes from a path.
    They list the ground truths of MTMC tracking in the MOTChallenge format 
    [frame, ID, left, top, width, height, 1, -1, -1, -1].
    Only frame, left, top, width and height are used.
    """
    bbs = []
    with open(path, 'r') as f:
        for line in f:
            # Split the line by commas
            bb = line.strip().split(',')
            # Extract values 0, 3, 4, 5, and 6
            values = [int(bb[0])] + [int(bb[i]) for i in range(2, 6)]
            bbs.append(values)
    return bbs

def load_frames(vdo, frame_index):
    """
    Loads frame of the video
    """

    # Open the video file
    cap = cv2.VideoCapture(vdo)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print('Error opening video file')
        exit()

    # Set the frame index for the next frame to retrieve
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame at the specified index
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print(f'Error reading frame {frame_index}')
        exit()

    # Convert the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Matplotlib
    #plt.imshow(frame)
    #plt.show()

    # Release the video file
    cap.release()
    return frame