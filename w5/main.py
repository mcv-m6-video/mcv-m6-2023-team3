# Import required packages
import motmetrics as mm
import train_features
from train_features import *
from utils import *


# This function computes the score for the given detections
def compute_score(det_info):
    # Initialize the MOT accumulator with automatic ID assignment
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

    return summary


def task2():
    # Set paths to the training data for ground truth
    gt_train_paths = ['aic19-track1-mtmc-train/train/S03', 'aic19-track1-mtmc-train/train/S04']

    # Set the base path for the videos
    base = 'aic19-track1-mtmc-train/train/S01'
    # Set the path for each video for each camera
    video_path = {
        'c001': "{}/c001/vdo.avi".format(base),
        'c002': "{}/c002/vdo.avi".format(base),
        'c003': "{}/c003/vdo.avi".format(base),
        'c004': "{}/c004/vdo.avi".format(base),
        'c005': "{}/c005/vdo.avi".format(base),
    }
    # Set the frame size for each camera
    frame_size = {
        'c001': [1920, 1080],
        'c002': [1920, 1080],
        'c003': [1920, 1080],
        'c004': [1920, 1080],
        'c005': [1280, 960],
    }

    # Set the list of camera names
    camerasList = ['c001', 'c002', 'c003', 'c004', 'c005']

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
        file = "detectionsS01_" + pickle_name + "_ssd512.pkl"
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
                    ft_vecCam1 = histogram_rgb(cropped_bboxes_cam1[n])
                    ft_vecCam2 = histogram_rgb(cropped_bboxes_cam2[n])
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
    summary = compute_score(det_info)
    print(summary)


task2()
