import pickle

import cv2
import numpy as np
import os

framesS03 = 'aic19-track1-mtmc-train/train/S04/'
camerasL = ['c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023', 'c024', 'c025', 'c026', 'c027', 'c028', 'c029', 'c030',
            'c031', 'c032', 'c033', 'c034', 'c035', 'c035', 'c036', 'c037', 'c038', 'c039', 'c040']


for cam in camerasL:
    print("start ", cam)
    path = framesS03 + cam + "/vdo.avi"
    # Open the video file
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        pathimage = framesS03 + cam + '/frames/'
        frame_number_str = str(i).zfill(5)
        cv2.imwrite(pathimage + f'{frame_number_str}.jpg', frame)
