# Import required packages
import os
import pickle
import sys


import numpy as np
import cv2
import imageio
from tqdm import trange
import motmetrics as mm
from sklearn.metrics.pairwise import pairwise_distances
from sort import Sort


from utils import Track, bb_intersection_over_union, refine_bbox, \
    Detection, format_pkl_train, reformat_predictions, centroid, \
    crop_image, histogram_hue, format_pkl
import train_features
from read_dataset import DataAnotations, group_by_frame


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
            centroits = np.array([[(d.xtl+d.xbr)/2, (d.ytl+d.ybr)/2] for d in track.detections])
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

def do_tracking(detector, dataset_path, distance_threshold, sequence, camera, save_path, use_kalman=False, generate_video=False, save_summary=False):
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
    reader = DataAnotations( dataset_path + '/' + sequence + '/' + camera + '/gt/gt.txt')
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
                detections.append(Detection(frame, int(detection[-1]), 'car',  *detection[:4]))
    
            detections_on_frame = detections

        # Do the tracking based on overlap
        tracks, addition_track_id = tracking_by_overlap(tracks, detections_on_frame ,addition_track_id)

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
        if generate_video and frame > end *0.75:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            for det in y_gt[frame]:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 6)

        frame_detections = []
        for det in detections.get(frame, []):
            frame_detections.append(det)
            if generate_video and frame > end *0.75:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)),  track.color, 6)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15),  track.color, -6)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 6)

        y_pred.append(frame_detections)

        if generate_video and frame > end *0.75 :
            writer.append_data(cv2.resize(img, (600, 350)))

        acc.update(
            [det.id for det in y_gt[frame]],
            [det.id for det in y_pred[-1]],
            compute_distance(frame, y_gt, y_pred)
        )
        save_tracks_txt = True
        if save_tracks_txt:
            filename = os.path.join(save_path, sequence + '_' + camera + "_"+ detector+"_"+"kalman"+'.txt')

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

# This function computes the score for the given detections
def compute_score(det_info):
    """
        Initialize the MOT accumulator with automatic ID assignment
        det_info(): Detections
    """
    acc = mm.MOTAccumulator(auto_id=True)

    # Define the ground truth files for each camera
    path_gttxts = ['aic19-track1-mtmc-train/train/S03/c010/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c011/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c012/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c013/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c014/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c015/gt/gt.txt']

    # Define the track names for each camera
    track_names = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']

    # Loop over each camera
    for cam_index in range(len(path_gttxts)):
        # Import the ground truth data for the current camera
        gt_info = train_features.import_gt_track(path_gttxts[cam_index])
        formatted_gt = format_pkl_train([gt_info], [track_names[cam_index]])
        cams_List, trackId_List, frames_List, boxes_List = reformat_predictions(formatted_gt)

        # Correct the ground truth data for the current camera
        camera_index = np.where(np.array(cams_List) == track_names[cam_index])
        camera_index = camera_index[0]
        new_frames = [frames_List[id] for id in camera_index]
        new_boxes = [boxes_List[id] for id in camera_index]
        new_tracksId = [trackId_List[id] for id in camera_index]

        # Correct the detection data for the current camera
        cam_idx_det = np.where(np.array(det_info[0]) == track_names[cam_index])
        cam_idx_det = cam_idx_det[0]
        new_frames_det = [det_info[2][id] for id in cam_idx_det]
        new_boxes_det = [det_info[3][id] for id in cam_idx_det]
        new_tracksId_det = [det_info[1][id] for id in cam_idx_det]

        # Loop over all frames in the current camera
        for frameID in trange(len(new_frames), desc="Score"):
            Nframe = new_frames[frameID]

            # Get the IDs of the tracks from the ground truth at this frame
            gt_list = [j for j, k in enumerate(new_frames) if k == Nframe]
            GTlist = [new_tracksId[i] for i in gt_list]
            GTbbox = [new_boxes[i] for i in gt_list]

            # Get the IDs of the detected tracks at this frame
            det_list = [j for j, k in enumerate(new_frames_det) if k == Nframe]
            DETlist = [new_tracksId_det[i] for i in det_list]
            detectedBoundingBox = [new_boxes_det[i] for i in det_list]

            # Compute the distance between each ground truth track and each detected track
            distances = []
            for i in range(len(GTlist)):
                dist = []
                # Compute the ground truth bounding box
                bboxGT = GTbbox[i]

                # Compute centroid GT
                centerGT = centroid(bboxGT)
                for j in range(len(DETlist)):
                    # Compute the predicted bbox
                    predictedBoundingBox = detectedBoundingBox[j]

                    # Compute centroid PR
                    centerPR = centroid(predictedBoundingBox)

                    # Compute euclidean distance
                    d = np.linalg.norm(np.array(centerGT) - np.array(centerPR))
                    dist.append(d)
                distances.append(dist)

            # Update the accumulator
            acc.update(GTlist, DETlist, distances)

    # Compute and show the final metric results
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr', 'precision', 'recall'], name='ACC:')
    print(summary)


def task2():
    """
        Task2 one about Multi-target Multiple-camera (MTSC) based on NCA + Track of the different three backbones ['mask_rcnn', 'ssd512', 'yolo3']
        This system is based on the AI City 19 dataset.
    """
    # Set paths to the training data for ground truth
    gt_train_paths = ['aic19-track1-mtmc-train/train/S01', 'aic19-track1-mtmc-train/train/S04']

    # Set the base path for the videos
    base = 'aic19-track1-mtmc-train/train/S03'

    # Set the path for each video for each camera
    video_path = {
        'c010': "{}/c010/vdo.avi".format(base),
        'c011': "{}/c011/vdo.avi".format(base),
        'c012': "{}/c012/vdo.avi".format(base),
        'c013': "{}/c013/vdo.avi".format(base),
        'c014': "{}/c014/vdo.avi".format(base),
        'c015': "{}/c015/vdo.avi".format(base),
    }
    # Set the frame size for each camera
    frame_size = {
        'c010': [1920, 1080],
        'c011': [2560, 1920],
        'c012': [2560, 1920],
        'c013': [2560, 1920],
        'c014': [1920, 1080],
        'c015': [1920, 1080]
    }

    # Set the list of camera names
    camerasList = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']

    # Set the list of pickle names
    pickle_names = ['c10', 'c11', 'c12', 'c13', 'c14', 'c15']

    # Set the flags
    video = False
    showVid = False

    # Create an empty list to store detections
    dets = []
    # Loop through each camera in the list
    for pickle_name in camerasList:
        # Set the path to the detection pickle file for this camera
        file = "detectionsS03_" + pickle_name + "_ssd512.pkl"
        print(file)
        # Open the pickle file and load the detections
        with open(file, 'rb') as f:
            dets.append(pickle.load(f))
            f.close()

    # Format the detections into a standard format
    detections = format_pkl(all_pkl=dets, camerasL=camerasList, isGt=False)

    """ Train """
    # Get ground truth information and video paths
    groundTruthData, video_paths = train_features.get_gt_info(gt_train_paths)

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
                    ft_vecCam1 = histogram_hue(cropped_bboxes_cam1[n])
                    ft_vecCam2 = histogram_hue(cropped_bboxes_cam2[n])
                    if ft_vecCam2 is None or ft_vecCam1 is None:
                        continue
                    formPairs = [np.vstack((ft_vecCam1, ft_vecCam2))]
                    formPairs = np.array(formPairs)
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

    cams_List, trackId_List, frames_List, boxes_List = reformat_predictions(corrected_detections)

    if video:
        # Generate colors for each track
        id_colors = []
        length = []
        for cc in range(len(camerasList)):
            length.append(len(corrected_detections[cc][camerasList[cc]]))
        max_tracks = np.max(length)

        for i in range(max_tracks):
            color = list(np.random.choice(range(256), size=3))
            id_colors.append(color)
        for cam in camerasList:
            # Define the codec and create VideoWriter object
            vidCapture = cv2.VideoCapture(video_path[cam])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(cam + '_task2.mp4', fourcc, 10.0, (frame_size[cam][0], frame_size[cam][1]))

            # filter the cam detections
            cam_idx = np.where(np.array(cams_List) == cam)
            cam_idx = cam_idx[0]
            new_frames = [frames_List[id] for id in cam_idx]
            new_boxes = [boxes_List[id] for id in cam_idx]
            new_tracksId = [trackId_List[id] for id in cam_idx]

            # for each frame draw rectangles to the detected bboxes
            for i, fr in enumerate(np.unique(new_frames)):
                idsFr = np.where(np.array(new_frames) == fr)
                idsFr = idsFr[0]
                vidCapture.set(cv2.CAP_PROP_POS_FRAMES, fr)
                im = vidCapture.read()[1]

                for j in range(len(idsFr)):
                    track_id = new_tracksId[idsFr[j]]
                    bbox = new_boxes[idsFr[j]]
                    color = id_colors[track_id]
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  (int(color[0]), int(color[1]), int(color[2])), 2)
                    cv2.putText(im, 'ID: ' + str(track_id), (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)

                if showVid:
                    cv2.imshow('Video', im)
                out.write(im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vidCapture.release()
            out.release()
            cv2.destroyAllWindows()

    # Compute the score for multitracking
    det_info = reformat_predictions(corrected_detections)
    compute_score(det_info)

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
        idf1s = do_tracking(detector, DATA_SET_PATH, THRESHOLD, SEQUENCE, CAMERA, SAVE_PATH, USE_KALMAN, GENERATE_VIDEO, SAVE_SUMMARY )
        all_idf1s.append(idf1s)
    print(all_idf1s)

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