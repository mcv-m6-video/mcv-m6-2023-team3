# Import required packages
import pickle

import cv2
from tqdm import trange

from compute_metric import *

# Function to perform overlap tracking
def overlapTracking(frameBoundingBoxes, video_path):
    # Define list to contain boxes per frame, id per frame and grab current frame
    boudningBoxes_per_frame = []
    id_per_frame = []
    frame = frameBoundingBoxes[0]

    # Loop over the frames
    for numFrames in trange(len(frameBoundingBoxes) - 1, desc="OverlapTracking"):
        # Grab next frame for detection
        next_frame = frameBoundingBoxes[numFrames + 1]

        # Assign a new ID to each unassigned bbox
        for i in range(len(frame)):
            # Append bounding box and ID
            boudningBoxes_per_frame.append([list(frame[i])])
            id_per_frame.append([numFrames])

        # Loop for each track, and we compute the iou with each detection of the next frame
        for id in range(len(boudningBoxes_per_frame)):
            length = len(boudningBoxes_per_frame[id])
            boundingBox_per_id = boudningBoxes_per_frame[id]
            bbox1 = boundingBox_per_id[length - 1]
            index_per_id = id_per_frame[id]

            # Initialize and update IOU for detection of next frame
            iou = []
            for detections in next_frame:
                # Detection of the next frame
                bbox2 = detections
                iou.append(frameIOU(np.array(bbox1), bbox2)[0])

            # Break if no more boxes
            if len(next_frame) == 0:
                break

            # Get best IOU for frame overlap
            overlap_iou = max(iou)

            # Threshold
            if overlap_iou > 0.5:
                best_detection = [j for j, k in enumerate(iou) if k == overlap_iou]
                best_detection = best_detection[0]

                # append to the list the bbox of the next frame
                boundingBox_per_id.append(list(next_frame[best_detection]))
                index_per_id.append(numFrames + 1)

                # Delete frame
                del next_frame[best_detection]

        # Update the frame
        frame = next_frame
