# Import required packages
import pickle
import motmetrics as mm
from tqdm import trange

from compute_metric import *
from read_data import *


# Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/blob/main/week3
# Get the bboxes
def centroid(box):  # box [x,y,w,h]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    x_center = (box[0] + x2) / 2
    y_center = (box[1] + y2) / 2
    return x_center, y_center


# Function to perform overlap tracking
def task2_1(path, video_path):
    frame_bboxes = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                frame_bboxes.append(pickle.load(openfile))
            except EOFError:
                break

    # Correct the data to the desired format
    aux_frame_boxes = []
    for frame_b in frame_bboxes:
        auxiliar = zip(*frame_b)
        aux_frame_boxes.append(list(auxiliar))
    frameBoundingBoxes = aux_frame_boxes

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

    _, gt = parse_annotations(path="ai_challenge_s03_c010-full_annotation.xml")

    # Initialize accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Loop for all frames
    for Nframe in trange(len(frame_bboxes), desc="Evaluation"):

        # get the ids of the tracks from the ground truth at this frame
        gt_list = [item[1] for item in gt if item[0] == Nframe]
        gt_list = np.unique(gt_list)

        # get the ids of the detected tracks at this frame
        pred_list = []
        for ID in range(len(id_per_frame)):
            aux = np.where(np.array(id_per_frame[ID]) == Nframe)[0]
            if len(aux) > 0:
                pred_list.append(int(ID))

        # Compute the distance for each pair
        distances = []
        for i in range(len(gt_list)):
            dist = []

            # Compute the ground truth bbox
            bboxGT = gt_list[i]
            bboxGT = [item[3:7] for item in gt if (item[0] == Nframe and item[1] == bboxGT)]
            bboxGT = list(bboxGT[0])

            # compute centroid GT
            centerGT = centroid(bboxGT)
            for j in range(len(pred_list)):
                # compute the predicted bbox
                bboxPR = pred_list[j]
                aux_id = id_per_frame[bboxPR].index(Nframe)
                bboxPR = boudningBoxes_per_frame[bboxPR][aux_id]

                # compute centroid PR
                centerPR = centroid(bboxPR)
                d = np.linalg.norm(np.array(centerGT) - np.array(centerPR))
                dist.append(d)
            distances.append(dist)

        # Update the accumulator
        acc.update(gt_list, pred_list, distances)

    # Compute and show the final metric results
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc:')
    print(summary)
    summary.to_csv("summary.csv")

