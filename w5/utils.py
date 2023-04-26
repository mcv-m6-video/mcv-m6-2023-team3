# Import required packages
import os
import pickle
import cv2
import numpy as np
import motmetrics as mm
from tqdm import trange
import train_features


# from https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class Track:
    def __init__(self, id, detections):
        self.id = id
        self.detections = detections
        self.terminated = False
        color = (list(np.random.choice(range(256), size=3)))
        self.color = (int(color[0]), int(color[1]), int(color[2]))

    def add_detection(self, detection):
        self.detections.append(detection)


class Detection:
    def __init__(self, frame, id, label, xtl, ytl, xbr, ybr):
        self.frame = frame
        self.id = id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.bbox = [self.xtl, self.ytl, self.xbr, self.ybr]
        self.width = abs(self.xbr - self.xtl)
        self.height = abs(self.ybr - self.ytl)
        self.area = self.width * self.height
        self.center = (int((self.xtl + self.xbr) / 2), int((self.ytl + self.ybr) / 2))


# from https://github.com/mcv-m6-video/mcv-m6-2020-team2
def refine_bbox(last_detections, new_detection, k=0.5):
    # No refinement for the first two frames
    if len(last_detections) < 2:
        return new_detection

    # Predict coordinates of new detection from last two detections
    pred_detection_xtl = 2 * last_detections[1].xtl - last_detections[0].xtl
    pred_detection_ytl = 2 * last_detections[1].ytl - last_detections[0].ytl
    pred_detection_xbr = 2 * last_detections[1].xbr - last_detections[0].xbr
    pred_detection_ybr = 2 * last_detections[1].ybr - last_detections[0].ybr

    # Compute average of predicted coordinates and detected coordinates
    refined_xtl = new_detection.xtl * k + pred_detection_xtl * (1 - k)
    refined_ytl = new_detection.ytl * k + pred_detection_ytl * (1 - k)
    refined_xbr = new_detection.xbr * k + pred_detection_xbr * (1 - k)
    refined_ybr = new_detection.ybr * k + pred_detection_ybr * (1 - k)

    # Get refined detection
    refined_detection = Detection(frame=new_detection.frame,
                                  id=new_detection.id,
                                  label=new_detection.label,
                                  xtl=refined_xtl,
                                  ytl=refined_ytl,
                                  xbr=refined_xbr,
                                  ybr=refined_ybr)

    return refined_detection


def histogram_hue(img):
    """
    Compute the histogram of hue values in an image.

    Args:
        img: An input image in the format of a numpy array.

    Returns:
        A 1D numpy array containing the histogram of hue values.
    """
    try:
        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except cv2.error:
        # Return None if there is an error in the conversion
        return None

    # Extract the hue channel from the HSV image
    h, _, _ = cv2.split(hsv)

    # Compute the histogram of hue values
    hist, _ = np.histogram(h, bins=864, range=(0, 180))

    # Reshape the histogram to a 1D numpy array
    return hist.reshape(-1)


# Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/blob/main/week5
def centroid(box):
    """
    Calculates the centroid coordinates of a bounding box.

    arduino
    Copy code
    Args:
        box (list): A bounding box in the format [x, y, w, h].

    Returns:
        tuple: The centroid coordinates (x,y) of the bounding box.
    """
    # calculate the coordinates of the bottom right corner of the bounding box
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]

    # calculate the coordinates of the centroid of the bounding box
    x_center = (box[0] + x2) / 2
    y_center = (box[1] + y2) / 2

    # Return centroid
    return x_center, y_center


def format_pkl_train(all_pkl, camerasL):
    """Format data from pickle files into a list of dictionaries.

        Args:
            all_pkl (list): List of dictionaries, each containing information about detections from a single camera.
            camerasL (list): List of camera names.

        Returns:
            list: A list of dictionaries, each containing information about detections from a single camera.

        """
    allDetections = []
    for i, cam in enumerate(camerasL):
        data = []
        for j, id in enumerate(all_pkl[i]['id']):
            detections = []
            list_frames = all_pkl[i]['frame'][j]

            # Remove last element of list_frames if it's -1
            if len(np.where(np.array(list_frames) == -1)[0]) > 0:
                del list_frames[-1]

            for k, frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]
                detections.append({'frame': frame, 'box': boxes})

            data.append({'track_id': id, 'info': detections})

        allDetections.append({cam: data})

    return allDetections


def format_gt_pkl(all_pkl, camerasL):
    """
       Format ground truth pickle data into a list of detections for each camera.

       Args:
           all_pkl (list): A list of dictionaries containing ground truth data for each camera.
           camerasL (list): A list of camera names.

       Returns:
           list: A list of dictionaries where each dictionary represents a camera and contains a list
           of dictionaries for each track ID in the camera. Each track ID dictionary contains the
           track ID and a list of dictionaries for each detection in the track. Each detection dictionary
           contains the frame number and the box coordinates for the detection.
       """
    allDetections = []

    # Iterate over each camera
    for i, cam in enumerate(camerasL):
        data = []

        # Iterate over each track ID in the current camera
        for j, id in enumerate(all_pkl[i]['id']):
            detections = []

            # Iterate over each frame in the current track ID
            list_frames = all_pkl[i]['frame'][j]

            # Remove the last frame if it is -1 (indicates the end of the track)
            if len(np.where(np.array(list_frames) == -1)[0]) > 0:
                del list_frames[-1]

            # Iterate over each box in the current track ID
            for k, frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]

                # Append the current frame and box to the detections list
                detections.append({'frame': frame, 'box': boxes})

            # Append the current track ID and detections to the data list
            data.append({'track_id': id, 'info': detections})

        # Append the current camera and data to the allDetections list
        allDetections.append({cam: data})

    # Return the list of all camera detections
    return allDetections


def import_gt_track(gt_path: str, outpath: str = 'gt_tracks.pkl', save: bool = True) -> dict:
    """
    This function reads a ground truth (gt) tracking file and returns a dictionary with the following format:
    {
        'id': [list of object ids],
        'frame': [list of frames for each object],
        'box': [list of bounding boxes for each object]
    }
    The gt file should be a comma-separated values (csv) file where each row represents a detection in the following format:
    frame_id, object_id, x1, y1, x2, y2

    Parameters:
        gt_path (str): The path to the ground truth tracking file.
        outpath (str, optional): The path where to save the output dictionary as a pickle file. Defaults to 'gt_tracks.pkl'.
        save (bool, optional): Whether to save the output dictionary or not. Defaults to True.

    Returns:
        dict: A dictionary with the format described above.
    """
    # Initialize variables
    data = {}
    ids = []
    frames = {}
    bboxes = {}

    # Read the gt tracking file line by line
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(",")

            # Extract object id, frame id, and bounding box coordinates
            obj_id = int(fields[1])
            frame_id = int(fields[0])
            bbox = [int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5])]

            # If the object id is already in the dictionary, add the frame and bounding box to its respective lists
            if obj_id in ids:
                frames[obj_id].append(frame_id)
                bboxes[obj_id].append(bbox)

            # Otherwise, create a new entry for the object id and initialize its frame and bounding box lists
            else:
                ids.append(obj_id)
                frames[obj_id] = [frame_id]
                bboxes[obj_id] = [bbox]

    # Combine all data into a single dictionary
    data['id'] = ids
    data['frame'] = list(frames.values())
    data['box'] = list(bboxes.values())

    # Save the dictionary as a pickle file if requested
    if save:
        with open(outpath, 'wb') as f:
            pickle.dump(data, f)

    # Return the data
    return data


def get_gt_info(gt_paths):
    """
    Get the ground truth information of multiple camera sequences.

    Args:
        gt_paths (list): A list of paths to the directories containing the ground truth data.

    Returns:
        A tuple containing:
        - data (dict): A dictionary containing the ground truth data organized by track ID, frame, box, and camera.
        - video_paths (dict): A dictionary containing the paths to the video files corresponding to each camera.
    """
    # Initialize variables
    camerasL = []  # List to store camera names
    data = {  # Dictionary to store ground truth data
        'id': [],  # List to store track IDs
        'frame': [],  # List to store frame numbers
        'box': [],  # List to store bounding boxes
        'cam': []  # List to store camera names
    }
    video_paths = {}  # Dictionary to store video paths

    # Iterate over the ground truth directories
    for seq_path in gt_paths:
        # Iterate over the camera directories in each ground truth directory
        for cam_path in os.listdir(seq_path):
            # Add the camera name to the list
            camerasL.append(cam_path)
            # Get the path to the ground truth file
            gt_file_path = os.path.join(seq_path, cam_path, 'gt', 'gt.txt')
            # Get the path to the video file
            video_paths[cam_path] = os.path.join(seq_path, cam_path, 'vdo.avi')
            # Import the ground truth data from the file
            gt_data = import_gt_track(gt_file_path)

            # Add the data to the main data dictionary
            data['id'].extend(gt_data['id'])
            data['frame'].extend(gt_data['frame'])
            data['box'].extend(gt_data['box'])
            data['cam'].extend([cam_path] * len(gt_data['id']))

    # Sort the data by track ID, frame, box, and camera
    id, fr, bo, ca = zip(*sorted(zip(data['id'], data['frame'], data['box'], data['cam'])))
    data['id'], data['frame'], data['box'], data['cam'] = list(id), list(fr), list(bo), list(ca)

    # Return the data and video paths as a tuple
    return data, video_paths


def crop_image(track, index_box_track):
    """
    Crop the region of interest from each frame of a given track.

    Parameters:
    track (dict): Dictionary containing information about a video track.
    index_box_track (list): List of indices representing the region of interest in each frame.

    Returns:
    croppedIms (list): List of cropped images.
    """
    # Initialize empty list to store the cropped images.
    croppedIms = []

    # Loop through each index in the index_box_track list.
    for i in range(len(index_box_track)):
        # Get the corresponding bounding box and frame path from the track information.
        id = index_box_track[i]
        bbox = track['info'][int(id * len(track['info']))]['box']
        bbox = [int(p) for p in bbox]
        path = track['info'][int(id * len(track['info']))]['frame_path']

        # Load the image from the given frame path and convert to RGB.
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop the region of interest from the image using the given bounding box coordinates.
        cropIm = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        # Append the cropped image to the list of cropped images.
        croppedIms.append(cropIm)

    # Return the list of cropped images.
    return croppedIms


def format_pkl(all_pkl, camerasL, isGt):
    # Define the path of the frames for camera S01
    framesS01 = 'aic19-track1-mtmc-train/train/S01'

    # Create empty lists to store detections information
    allDetections = []  # List of dictionaries containing camera ID and track IDs with corresponding detections
    boxes_List = []  # List of bounding box coordinates for each detection
    frames_List = []  # List of frame numbers for each detection
    trackId_List = []  # List of track IDs for each detection
    cams_List = []  # List of camera IDs for each detection

    # Iterate over each camera and its corresponding detections
    for i, cam in enumerate(camerasL):
        data = []  # List of dictionaries containing track ID and corresponding detections for the current camera
        for j, id in enumerate(all_pkl[i]['id']):
            detections = []  # List of dictionaries containing detection information for the current track ID
            list_frames = all_pkl[i]['frame'][j]

            # If the last frame number in the list is -1, remove it as it represents a non-existing frame
            if len(np.where(np.array(list_frames) == -1)[0]) > 0:
                del list_frames[-1]

            # Iterate over each frame number and its corresponding bounding box coordinates for the current track ID
            for k, frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]  # Bounding box coordinates for the current detection
                cams_List.append(cam)  # Append the camera ID for the current detection to the cameras list
                trackId_List.append(id)  # Append the track ID for the current detection to the track IDs list
                frames_List.append(frame)  # Append the frame number for the current detection to the frames list
                boxes_List.append(
                    boxes)  # Append the bounding box coordinates for the current detection to the boxes list

                # If isGt is False, add the path of the corresponding frame image to the detection information
                if not isGt:
                    frame_path = "{}/frames/{}.jpg".format(os.path.join(framesS01, cam), str(frame).zfill(5))
                    detections.append({'frame': frame, 'frame_path': frame_path, 'box': boxes})
                # Otherwise, only add the bounding box coordinates to the detection information
                else:
                    detections.append({'frame': frame, 'box': boxes})

            # Append the track ID and corresponding detections to the data list
            data.append({'track_id': id, 'info': detections})

        # Add the camera ID and corresponding data (track IDs with corresponding detections) to the allDetections list
        allDetections.append({cam: data})

    return allDetections


def reformat_predictions(correct_predictions):
    """
    Reformat correctly predicted detections into separate lists for camera names, track IDs, frame numbers, and
    detection boxes, sorted by frame number.

    Args:
    - correct_pred (list): A list of correctly predicted detections in the format outputted by the tracking algorithm.

    Returns:
    - cams_list (list): A list of camera names.
    - track_id_list (list): A list of track IDs.
    - frames_list (list): A list of frame numbers.
    - boxes_list (list): A list of detection boxes.
    """
    # initialize lists for storing reformatted predictions
    boxes_list = []
    frames_list = []
    track_id_list = []
    cams_list = []

    # iterate through each prediction
    for i in range(len(correct_predictions)):
        # get the camera name
        cam = list(correct_predictions[i].keys())[0]
        # iterate through each track in the camera
        for j in range(len(correct_predictions[i][cam])):
            # get the track id
            track_id = correct_predictions[i][cam][j]['track_id']

            # iterate through each detection in the track
            for k in range(len(correct_predictions[i][cam][j]['info'])):
                # get the frame number and detection box
                frame = correct_predictions[i][cam][j]['info'][k]['frame']
                boxes = correct_predictions[i][cam][j]['info'][k]['box']

                # append the camera name, track id, frame number, and detection box to their respective lists
                cams_list.append(cam)
                track_id_list.append(track_id)
                frames_list.append(frame)
                boxes_list.append(boxes)

    # sort the lists by frame number
    frames_list, boxes_list, track_id_list, cams_list = zip(
        *sorted(zip(frames_list, boxes_list, track_id_list, cams_list)))

    # return the reformatted predictions as lists
    return list(cams_list), list(track_id_list), list(frames_list), list(boxes_list)


def compute_score(det_info):
    """
        Initialize the MOT accumulator with automatic ID assignment
        det_info(): Detections
    """
    acc = mm.MOTAccumulator(auto_id=True)

    # Define the ground truth files for each camera
    path_gttxts = ['aic19-track1-mtmc-train/train/S01/c001/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S01/c002/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S01/c003/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S01/c004/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S01/c005/gt/gt.txt']

    # Define the track names for each camera
    track_names = ['c001', 'c002', 'c003', 'c004', 'c005']

    # Loop over each camera
    for cam_index in range(len(path_gttxts)):
        # Import the ground truth data for the current camera
        gt_info = train_features.import_gt_track(path_gttxts[cam_index])
        formatted_gt = format_pkl_train([gt_info], [track_names[cam_index]])
        camera_List, trackId_List, frames_List, boxes_List = reformat_predictions(formatted_gt)

        # Correct the ground truth data for the current camera
        camera_index = np.where(np.array(camera_List) == track_names[cam_index])
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
