import os
from glob import glob
from copy import deepcopy
import random


import numpy as np
import cv2
import imageio
from tqdm import trange
from matplotlib import pyplot as plt
import motmetrics as mm


from utils import Track, bb_intersection_over_union, refine_bbox, Detection
from read_dataset import DataAnotations, group_by_frame
from sklearn.metrics.pairwise import pairwise_distances
from sort import Sort


GENERATE_VIDEO = False
SAVE_SUMMARY = True
SAVE_PATH = 'results/week5/task_1'
THRESHOLD = 675
SEQUENCE = 'S03'
CAMERA = 'c010'
DATA_SET_PATH = "aic19/train"
DETECTORS = ['mask_rcnn', 'ssd512', 'yolo3']


def compare_box_detection(last_detection, additional_detections):
    """
    Check for all the additional detections that has not matched with a track if they match for the curren track detection.
    """
    max_iou = 0
    for detection in additional_detections:
        iou = bb_intersection_over_union(last_detection.bbox, detection.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = detection
    if max_iou > 0:
        best_match.id = last_detection.id
        return best_match
    else:
        return None


## TO BE REPLACE by ours
def remove_static_tracks(tracks, distance_threshold, min_track_len=5):
    new_tracks = []
    for track in tracks:
        if len(track.detections) > min_track_len:
            centroids_of_detections = np.array([[(d.xtl+d.xbr)/2, (d.ytl+d.ybr)/2] for d in track.detections])
            dists = pairwise_distances(centroids_of_detections, centroids_of_detections, metric='euclidean')

            if np.max(dists) > distance_threshold:
                new_tracks.append(track)

    return new_tracks


def update_tracks_by_overlap(tracks, new_detections, addition_track_id):
    for track in tracks:
        if track.terminated:
            continue

        # Compare track detection in last frame with new detections
        following_box = compare_box_detection(track.detections[-1], new_detections)
        # If there's a match, refine detections
        if following_box:
            refined_detection = refine_bbox(track.detections[-2:], following_box)
            track.add_detection(refined_detection)
            new_detections.remove(following_box)
        else:
            track.terminated = True

    # Update tracks with unused detections after matching
    for additional_detections in new_detections:
        additional_detections.id = addition_track_id + 1
        new_track = Track(addition_track_id + 1, [additional_detections])
        tracks.append(new_track)
        addition_track_id += 1

    return tracks, addition_track_id


def compute_distance(frame, y_gt, y_pred):
    X = np.array([det.center for det in y_gt[frame]])
    Y = np.array([det.center for det in y_pred[-1]])

    try:
        dists = pairwise_distances(X, Y, metric='euclidean')
    except ValueError:
        dists = np.array([])
    return dists

def do_tracking(detector, distance_threshold, sequence, camera, save_path, use_kalman=False):

    # Load the ground thruth
    reader = DataAnotations( DATA_SET_PATH + '/' + sequence + '/' + camera + '/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'])

    # Load the boxes gathered from the detector
    reader = DataAnotations(DATA_SET_PATH + '/' + sequence + '/' + camera + '/det/det_' + detector + '.txt')
    dets = reader.get_annotations(classes=['car'])

    # Load the video test to perform the inference
    cap = cv2.VideoCapture(DATA_SET_PATH + '/' + sequence + '/' + camera + '/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    os.makedirs(save_path, exist_ok=True)
    if GENERATE_VIDEO:
        writer = imageio.get_writer(
            os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '.gif'), fps=25
        )

    if use_kalman:
        tracker = Sort()
    y_gt = []
    tracks = []
    addition_track_id = 0  # Counter to give id to those detections that become a track
    start = 0
    end = int(n_frames * 0.5)

    for frame in trange(start, end, desc='Tracking'):

        detections_on_frame_ = dets.get(frame, [])
        detections_on_frame = []
        for d in detections_on_frame_:
            detections_on_frame.append(d)

        if use_kalman:
                
            bounding_boxes = []
            for detection in detections_on_frame:
                bbox = [*detection.bbox, 1]
                bounding_boxes.append(bbox)
    
            # Update the tracker with the array of bounding boxes
            detections_on_frame = tracker.update(np.array(bounding_boxes))            

            detections = []
            for detection in detections_on_frame:
                detections.append(Detection(frame, int(detection[-1]), 'car',  *detection[:4]))
    
            # Update the detections_on_frame list with the list of Detection objects
            detections_on_frame = detections


        tracks, addition_track_id = update_tracks_by_overlap(tracks, detections_on_frame ,addition_track_id)

        y_gt.append(gt.get(frame, []))

    idf1s = []

    acc = mm.MOTAccumulator(auto_id=True)
    y_pred = []

    moving_tracks = remove_static_tracks(tracks, distance_threshold)
    detections = []
    for track in moving_tracks:
        detections.extend(track.detections)
    detections = group_by_frame(detections)

    for frame in trange(start, end, desc='Accumulating detections'):

        if GENERATE_VIDEO and frame > end *0.75:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            for det in y_gt[frame]:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 6)

        frame_detections = []
        for det in detections.get(frame, []):
            frame_detections.append(det)
            if GENERATE_VIDEO and frame > end *0.75:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)),  track.color, 6)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15),  track.color, -6)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 6)

        y_pred.append(frame_detections)

        if GENERATE_VIDEO and frame > end *0.75 :
            writer.append_data(cv2.resize(img, (600, 350)))

        acc.update(
            [det.id for det in y_gt[frame]],
            [det.id for det in y_pred[-1]],
            compute_distance(frame, y_gt, y_pred)
        )

    print('Additional metrics:')
    # compute the metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'precision', 'recall'], name='acc')
    print(summary)

    if SAVE_SUMMARY:
        with open(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '_' + str(
                distance_threshold) + '.txt'), 'w') as f:
            f.write(str(summary))

    idf1s.append(summary['idf1']['acc'])

    cv2.destroyAllWindows()
    if GENERATE_VIDEO:
        writer.close()

    return idf1s

all_idf1s = []
for detector in DETECTORS:
    print("Detector:" + detector)
    idf1s = do_tracking(detector, THRESHOLD, SEQUENCE, CAMERA, SAVE_PATH)

    all_idf1s.append(idf1s)

i=0
plt.bar(detector, all_idf1s[i])
plt.xlabel('detectors')
plt.ylabel('IDF1')
if SAVE_PATH:
    plt.savefig(os.path.join(SAVE_PATH, 'task1_' + SEQUENCE + '_' + CAMERA + '_' + 'dist-th_vs_idf1.png'))
