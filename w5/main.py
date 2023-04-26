# Import required packages
import os
import pickle
import sys

import cv2
import imageio
import motmetrics as mm
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import trange

import train_features
from read_dataset import DataAnotations, group_by_frame
from sort import Sort
from utils import Track, bb_intersection_over_union, refine_bbox, \
    Detection, reformat_predictions, crop_image, histogram_hue, format_pkl, compute_score, get_gt_info


def compare_box_detection(last_detection, additional_detections):
    """
        Check for all the additional detections that has not matched with a track if they match for the curren track detection.
        last_detection(Detection): last detection inside the track
        additional_detections(List[Detections]): Additional matches to compare
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


def rm_parked_cars(tracks, threshold, min_len=5):
    """
        Function to remove the cars parked from the detections
        tracks(list[Tracks]): list of track objects containing the detections
        threshold(int): threshold value that sets the minimum distance to be considered tacks
        min_len(int): min number of frames
    """
    addtional_track = []
    for track in tracks:
        if len(track.detections) > min_len:
            centroits = np.array([[(d.xtl + d.xbr) / 2, (d.ytl + d.ybr) / 2] for d in track.detections])
            distance = pairwise_distances(centroits, centroits, metric='euclidean')

            if np.max(distance) > threshold:
                addtional_track.append(track)

    return addtional_track


def tracking_by_overlap(tracks, new_detections, addition_track_id):
    """
        Function to perform the tracking by overlap strategy
        tracks(List[Tracks]): list of track objects containing the detections
        new_detections(List[Detections]): list of Detections that has not been considered inside any Track
        addition_track_id(int): Num of additional tracks
    """
    for track in tracks:
        if track.terminated:
            continue

        following_box = compare_box_detection(track.detections[-1], new_detections)
        if following_box:
            refined_detection = refine_bbox(track.detections[-2:], following_box)
            track.add_detection(refined_detection)
            new_detections.remove(following_box)
        else:
            track.terminated = True

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


def do_tracking(detector, dataset_path, distance_threshold, sequence, camera, save_path, use_kalman=False,
                generate_video=False, save_summary=False):
    """
        Main function to perform the detection based on tracking by overlap and that can be optionally be based on kalman filter tacker too
        detector(str): The name of the detector use to perform the object detection ['mask_rcnn', 'ssd512', 'yolo3']
        distance_threshold(int): threshold value that sets the minimum distance to be considered tacks
        sequence(str): The name of the sequence to be evaluated
        camera(str): The name of the camera to be evaluated
        save_path(str): The path where to save the results
        use_kalman(bool): Bool var too indicate if use the kalman tracker or not.
    """

    # Load the ground thruth
    reader = DataAnotations(dataset_path + '/' + sequence + '/' + camera + '/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'])

    # Load the boxes gathered from the detector
    reader = DataAnotations(dataset_path + '/' + sequence + '/' + camera + '/det/det_' + detector + '.txt')
    dets = reader.get_annotations(classes=['car'])

    # Load the video test to perform the inference
    cap = cv2.VideoCapture(dataset_path + '/' + sequence + '/' + camera + '/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(save_path, exist_ok=True)
    if generate_video:
        writer = imageio.get_writer(
            os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '.gif'), fps=25
        )

    if use_kalman:
        tracker = Sort()
    y_gt = []
    tracks = []
    addition_track_id = 0  # Counter to give id to those detections that become a track
    start = 0
    end = int(n_frames * 1)
    for frame in trange(start, end, desc='Tracking'):

        detections_on_frame_ = dets.get(frame, [])
        detections_on_frame = []
        for d in detections_on_frame_:
            detections_on_frame.append(d)

        # Use kalman to do the tracking
        if use_kalman:

            bounding_boxes = []
            for detection in detections_on_frame:
                bbox = [*detection.bbox, 1]
                bounding_boxes.append(bbox)

            # Update the tracker with the array of bounding boxes
            detections_on_frame = tracker.update(np.array(bounding_boxes))

            detections = []
            for detection in detections_on_frame:
                detections.append(Detection(frame, int(detection[-1]), 'car', *detection[:4]))

            detections_on_frame = detections

        # Do the tracking based on overlap
        tracks, addition_track_id = tracking_by_overlap(tracks, detections_on_frame, addition_track_id)

        y_gt.append(gt.get(frame, []))

    idf1s = []

    acc = mm.MOTAccumulator(auto_id=True)
    y_pred = []

    # Function to remove the parked cars from the detection
    moving_tracks = rm_parked_cars(tracks, distance_threshold)
    detections = []
    for track in moving_tracks:
        detections.extend(track.detections)
    detections = group_by_frame(detections)

    for frame in trange(start, end):

        # Generate a small gift for the slides
        if generate_video and frame > end * 0.75:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            for det in y_gt[frame]:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 6)

        frame_detections = []
        for det in detections.get(frame, []):
            frame_detections.append(det)
            if generate_video and frame > end * 0.75:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 6)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15), track.color, -6)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 6)

        y_pred.append(frame_detections)

        if generate_video and frame > end * 0.75:
            writer.append_data(cv2.resize(img, (600, 350)))

        acc.update(
            [det.id for det in y_gt[frame]],
            [det.id for det in y_pred[-1]],
            compute_distance(frame, y_gt, y_pred)
        )
        save_tracks_txt = True
        if save_tracks_txt:
            filename = os.path.join(save_path, sequence + '_' + camera + "_" + detector + "_" + "kalman" + '.txt')

            lines = []
            for track in moving_tracks:
                for det in track.detections:
                    lines.append(
                        (det.frame, track.id, det.xtl, det.ytl, det.width, det.height, "1", "-1", "-1", "-1"))

            lines = sorted(lines, key=lambda x: x[0])
            with open(filename, "w") as file:
                for line in lines:
                    file.write(",".join(list(map(str, line))) + "\n")

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp'], name='acc')
    print(summary)

    if save_summary:
        with open(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '_' + str(
                distance_threshold) + '.txt'), 'w') as f:
            f.write(str(summary))

    idf1s.append(summary['idf1']['acc'])

    cv2.destroyAllWindows()
    if generate_video:
        writer.close()

    return idf1s


def task1():
    """
        Task1 one about Multi-target single-camera (MTSC) tracking by overlap and Kalman.
        There are thre possible backbones ['mask_rcnn', 'ssd512', 'yolo3']
        This system is based on the AI City 19 dataset.
    """

    GENERATE_VIDEO = False
    SAVE_SUMMARY = True
    SAVE_PATH = 'results/week5/task_1'
    THRESHOLD = 675
    SEQUENCE = 'S03'
    CAMERA = 'c010'
    DATA_SET_PATH = "aic19/train"
    DETECTORS = ['mask_rcnn', 'ssd512', 'yolo3']
    USE_KALMAN = False
    all_idf1s = []
    # For all the three detector compute the idf1s
    for detector in DETECTORS:
        print("Detector:" + detector)
        idf1s = do_tracking(detector, DATA_SET_PATH, THRESHOLD, SEQUENCE, CAMERA, SAVE_PATH, USE_KALMAN, GENERATE_VIDEO,
                            SAVE_SUMMARY)
        all_idf1s.append(idf1s)
    print(all_idf1s)


def task2():
    """
        Task2 one about Multi-target Multiple-camera (MTSC) based on NCA + Track of the different three backbones ['mask_rcnn', 'ssd512', 'yolo3']
        This system is based on the AI City 19 dataset.
    """
    # Set paths to the training data for ground truth
    gt_train_paths = ['aic19-track1-mtmc-train/train/S03', 'aic19-track1-mtmc-train/train/S04']

    # Set the base path for the videos
    base = 'aic19-track1-mtmc-train/train/S01'

    # Set the list of camera names
    camerasList = ['c001', 'c002', 'c003', 'c004', 'c005']

    # Create an empty list to store detections
    detections = []

    # Loop through each camera in the list
    for pickle_name in camerasList:
        # Set the path to the detection pickle file for this camera
        file = "detectionsS01_" + pickle_name + "_ssd512.pkl"
        print(file)
        # Open the pickle file and load the detections
        with open(file, 'rb') as f:
            detections.append(pickle.load(f))
            f.close()

    # Format the detections into a standard format
    detections = format_pkl(all_pkl=detections, camerasL=camerasList, isGt=False)

    """ Train """
    # Get ground truth information and video paths
    groundTruthData, video_paths = get_gt_info(gt_train_paths)

    # Train the classifier
    nca = train_features.train(groundTruthData, video_paths)

    # Set the indices of the boxes to crop for each track
    index_box_track = [0.25, 0.5, 0.60]

    # Set the count of cameras to 1
    count_cams = 1

    # Create list to store scores
    total_scores = []
    corrected_detections = detections

    # Loop through each camera in the detections
    for idCam in trange(len(detections) - 1):
        # Get the current camera and the next camera in the list
        cam1 = camerasList[idCam]
        cam2 = camerasList[count_cams]

        # Increase the count of cameras
        count_cams += 1

        # Get the number of tracks in each camera
        num_tracks1 = len(detections[idCam][cam1])
        num_tracks2 = len(detections[idCam + 1][cam2])

        # Loop through each track in the current camera
        for i in range(num_tracks1):
            # Get the current track and crop the image to the bounding box
            track1 = detections[idCam][cam1][i]
            cropped_bboxes_cam1 = crop_image(track1, index_box_track)

            # Loop through each track in the next camera
            for j in range(num_tracks2):
                # Get the current track and crop the image to the bounding box
                track2 = detections[idCam + 1][cam2][j]
                cropped_bboxes_cam2 = crop_image(track2, index_box_track)

                # Calculate the score between the two cropped images
                scores = []
                for n in range(len(cropped_bboxes_cam1)):
                    featuresCamera1 = histogram_hue(cropped_bboxes_cam1[n])
                    featuresCamera2 = histogram_hue(cropped_bboxes_cam2[n])

                    # Check if features are none
                    if featuresCamera2 is None or featuresCamera1 is None:
                        continue

                    # Stack features to form pairs
                    formPairs = [np.vstack((featuresCamera1, featuresCamera2))]
                    formPairs = np.array(formPairs)

                    # Calculate scores
                    score = nca.score_pairs(formPairs).mean()
                    scores.append(score)

                # Get the mean score for the two tracks
                mean_score = np.mean(np.array(scores))
                total_scores.append(scores)

                # If the score is lower than 50, update the track ID in the next camera
                if mean_score < 50:
                    track2['track_id'] = track1['track_id']

                    # Update the detections
                    corrected_detections[idCam][cam1][i] = track1
                    corrected_detections[idCam + 1][cam2][j] = track2

    # Compute the score for multitracking
    detection_info = reformat_predictions(corrected_detections)
    compute_score(detection_info)


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1

    if task == 1:
        task1()
    elif task == 2:
        task2()


if __name__ == "__main__":
    main(sys.argv)
