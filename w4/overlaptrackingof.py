# Import required packages
import pickle

import cv2
import motmetrics as mm
from tqdm import trange
from block_matching import compute_block_matching
from compute_metric import *
from read_data import *


# Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/blob/main/week4
# Get the bboxes
def centroid(box):
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    x_center = (box[0] + x2) / 2
    y_center = (box[1] + y2) / 2
    return x_center, y_center


# Function to perform overlap tracking
def task2_1(path="boxesScores.pkl", video_path= '/Users/advaitdixit/Documents/Masters/mcv-m6-2023-team3/AICity_data/train/S03/c010/01_vdo.avi'):
    pkl_path = path
    video_path = video_path
    showVid = True

    # Get the bboxes
    frame_boundingboxes = []
    with (open(pkl_path, "rb")) as openfile:
        while True:
            try:
                frame_boundingboxes.append(pickle.load(openfile))
            except EOFError:
                break

    frame_boundingboxes = frame_boundingboxes[0]

    # correct the data to the desired format
    aux_frame_boxes = []
    for frame_b in frame_boundingboxes:
        auxiliar, _ = zip(*frame_b)
        aux_frame_boxes.append(list(auxiliar))
    frame_boundingboxes = aux_frame_boxes

    # Once we have done the detection we can start with the tracking
    cap = cv2.VideoCapture(video_path)
    previous_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    boudningBoxes_per_frame = []
    id_per_frame = []
    frame = frame_boundingboxes[0]  # load the bbox for the first frame

    # Since we evaluate the current frame and the consecutive, we loop for range - 1
    for Nframe in trange(len(frame_boundingboxes) - 1, desc="Overlap Tracking with Optical Flow"):
        next_frame = frame_boundingboxes[Nframe + 1]
        current_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

        # apply optical flow to improve the bounding box and get better iou with the following frame
        # predict flow with block matching
        predicted_flow = compute_block_matching(previous_frame, current_frame, 'backward', searchArea=96,
                                                blockSize=16, method="cv2.TM_CCORR_NORMED", q=16)

        # assign a new ID to each unassigned bbox
        for i in range(len(frame)):
            new_bbox = frame[i]

            # append the bbox to the list
            boundingBox_per_id = [list(new_bbox)]
            boudningBoxes_per_frame.append(boundingBox_per_id)

            # append the id to the list
            index_per_id = [Nframe]
            id_per_frame.append(index_per_id)

        # we loop for each track and we compute the iou with each detection of the next frame
        for id in range(len(boudningBoxes_per_frame)):
            length = len(boudningBoxes_per_frame[id])
            boundingBox_per_id = boudningBoxes_per_frame[id]
            bbox1 = boundingBox_per_id[length - 1]
            index_per_id = id_per_frame[id]

            vectorU = predicted_flow[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2]), 0]
            vectorV = predicted_flow[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2]), 1]
            dx = vectorU.mean()
            dy = vectorV.mean()
            
            # apply movement to the bbox
            new_bbox1 = list(np.zeros(4))
            new_bbox1[0] = bbox1[0] + dx
            new_bbox1[2] = bbox1[2] + dx
            new_bbox1[1] = bbox1[1] + dy
            new_bbox1[3] = bbox1[3] + dy

            # get the list of ious, one with each detection of the next frame
            iou = []
            for detections in range(len(next_frame)):
                bbox2 = next_frame[detections]  # detection of the next frame
                iou.append(frameIOU(np.array(new_bbox1), bbox2)[0])

            # break the loop if there are no more bboxes in the frame to track
            if len(next_frame) == 0:
                break

            # assign the bbox to the closest track
            overlap_iou = max(iou)

            # if the mas iou is lower than 0.5, we assume that it doesn't have a correspondence
            if overlap_iou > 0.5:
                best_detection = [j for j, k in enumerate(iou) if k == overlap_iou]
                best_detection = best_detection[0]

                # append to the list the bbox of the next frame
                boundingBox_per_id.append(list(next_frame[best_detection]))
                index_per_id.append(Nframe + 1)

                # we delete the detection from the list in order to speed up the following comparisons
                del next_frame[best_detection]

        frame = next_frame  # the next frame will be the current
        previous_frame = current_frame  # update the frame for next iteration

    # Generate colors for each track
    id_colors = []
    for i in range(len(id_per_frame)):
        color = list(np.random.choice(range(256), size=3))
        id_colors.append(color)

    # Define the codec and create VideoWriter object
    vidCapture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('of.mp4', fourcc, 10.0, (1920, 1080))

    # for each frame draw rectangles to the detected bboxes
    for i in trange(len(frame_boundingboxes), desc="Video"):
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
                cv2.putText(im, 'ID: ' + str(id), (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)
        if showVid:
            cv2.imshow('Video', im)
        out.write(im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidCapture.release()
    out.release()
    cv2.destroyAllWindows()

    # Load gt for plot
    _, gt = parse_annotations(path="ai_challenge_s03_c010-full_annotation.xml")

    # init accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Loop for all frames
    for Nframe in trange(len(frame_boundingboxes), desc="Evaluation"):

        # get the ids of the tracks from the ground truth at this frame
        gt_list = [item[1] for item in gt if item[0] == Nframe]
        gt_list = np.unique(gt_list)

        # get the ids of the detected tracks at this frame
        prediction_list = []
        for ID in range(len(id_per_frame)):
            aux = np.where(np.array(id_per_frame[ID]) == Nframe)[0]
            if len(aux) > 0:
                prediction_list.append(int(ID))

        # compute the distance for each pair
        distances = []
        for i in range(len(gt_list)):
            dist = []
            # compute the ground truth bbox
            bboxGT = gt_list[i]
            bboxGT = [item[3:7] for item in gt if (item[0] == Nframe and item[1] == bboxGT)]
            bboxGT = list(bboxGT[0])

            # compute centroid GT
            centerGT = centroid(bboxGT)
            for j in range(len(prediction_list)):
                # compute the predicted bbox
                bboxPR = prediction_list[j]
                aux_id = id_per_frame[bboxPR].index(Nframe)
                bboxPR = boudningBoxes_per_frame[bboxPR][aux_id]

                # compute centroid PR
                centerPR = centroid(bboxPR)
                d = np.linalg.norm(np.array(centerGT) - np.array(centerPR))  # euclidean distance
                dist.append(d)
            distances.append(dist)

        # update the accumulator
        acc.update(gt_list, prediction_list, distances)

    # Compute and show the final metric results
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='metrics')
    print(summary)
    summary.to_csv("summary.csv")


task2_1()
