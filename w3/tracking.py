# Import required packages
import pickle
import cv2
from tqdm import trange
from compute_metric import *

# Get the bboxes
frame_bboxes = []
with (open("boxesScores1.pkl", "rb")) as openfile:
    while True:
        try:
            frame_bboxes.append(pickle.load(openfile))
        except EOFError:
            break
frame_bboxes = frame_bboxes[0]

# correct the data to the desired format
aux_frame_boxes = []
for frame_b in frame_bboxes:
    auxiliar, _ = zip(*frame_b)
    aux_frame_boxes.append(list(auxiliar))
frame_bboxes = aux_frame_boxes


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

    id_colors = []
    for i in range(len(id_per_frame)):
        color = list(np.random.choice(range(256), size=3))
        id_colors.append(color)

    # Define the codec and create VideoWriter object
    vidCapture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('overlap_tracking.mp4', fourcc, 10.0, (1920, 1080))
    # for each frame draw rectangles to the detected bboxes
    for i in trange(len(frameBoundingBoxes), desc="Video"):
        vidCapture.set(cv2.CAP_PROP_POS_FRAMES, i)
        im = vidCapture.read()[1]
        for id in range(len(id_per_frame)):
            ids = id_per_frame[id]
            if i in ids:
                id_index = ids.index(i)
                bbox = boudningBoxes_per_frame[id][id_index]
                color = id_colors[id]
                cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              (int(color[0]), int(color[1]), int(color[2])), 2)
                cv2.putText(im, 'Object ID: ' + str(id), (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)
        cv2.imshow('Video', im)
        out.write(im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidCapture.release()
    out.release()
    cv2.destroyAllWindows()


overlapTracking(frame_bboxes, video_path="../AICity_data/train/S03/c010/01_vdo.avi")
