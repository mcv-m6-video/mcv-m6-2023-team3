# Import required packages
import motmetrics as mm
import train_features
from train_features import *
from utils import *


def compute_score(det_info):
    # init accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    path_gttxts = ['aic19-track1-mtmc-train/train/S03/c010/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c011/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c012/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c013/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c014/gt/gt.txt',
                   'aic19-track1-mtmc-train/train/S03/c015/gt/gt.txt']

    track_names = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']

    for cam_index in range(len(path_gttxts)):

        gt_info = train_features.import_gt_track(path_gttxts[cam_index])
        b = train_features.format_pkl([gt_info], [track_names[cam_index]])
        cams_List, trackId_List, frames_List, boxes_List = reformat_predictions(b)

        # gt correction
        cam_idx = np.where(np.array(cams_List) == track_names[cam_index])
        cam_idx = cam_idx[0]
        new_frames = [frames_List[id] for id in cam_idx]
        new_boxes = [boxes_List[id] for id in cam_idx]
        new_tracksId = [trackId_List[id] for id in cam_idx]

        # det correction
        cam_idx_det = np.where(np.array(det_info[0]) == track_names[cam_index])
        cam_idx_det = cam_idx_det[0]
        new_frames_det = [det_info[2][id] for id in cam_idx_det]
        new_boxes_det = [det_info[3][id] for id in cam_idx_det]
        new_tracksId_det = [det_info[1][id] for id in cam_idx_det]

        # Loop for all frames
        for frameID in trange(len(new_frames), desc="Score"):
            Nframe = new_frames[frameID]

            # get the ids of the tracks from the ground truth at this frame
            gt_list = [j for j, k in enumerate(new_frames) if k == Nframe]
            GTlist = [new_tracksId[i] for i in gt_list]
            GTbbox = [new_boxes[i] for i in gt_list]

            # get the ids of the detected tracks at this frame
            det_list = [j for j, k in enumerate(new_frames_det) if k == Nframe]
            DETlist = [new_tracksId_det[i] for i in det_list]
            DETbbox = [new_boxes_det[i] for i in det_list]

            # compute the distance for each pair
            distances = []
            for i in range(len(GTlist)):
                dist = []
                # compute the ground truth bbox
                bboxGT = GTbbox[i]
                # compute centroid GT
                centerGT = centroid(bboxGT)
                for j in range(len(DETlist)):
                    # compute the predicted bbox
                    bboxPR = DETbbox[j]
                    # compute centroid PR
                    centerPR = centroid(bboxPR)
                    d = np.linalg.norm(np.array(centerGT) - np.array(centerPR))  # euclidean distance
                    dist.append(d)
                distances.append(dist)

            # update the accumulator
            acc.update(GTlist, DETlist, distances)

    # Compute and show the final metric results
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr', 'precision', 'recall'], name='ACC:')

def task2():
    # Relative paths
    gt_train_paths = ['aic19-track1-mtmc-train/train/S03', 'aic19-track1-mtmc-train/train/S04']
    fps_r = {
        'c010': 1.0,
        'c011': 1.0,
        'c012': 1.0,
        'c013': 1.0,
        'c014': 1.0,
        'c015': 10.0 / 8.0,
    }
    timestamp = {
        'c010': 8.715,
        'c011': 8.457,
        'c012': 5.879,
        'c013': 0,
        'c014': 5.042,
        'c015': 8.492,
    }
    base = 'aic19-track1-mtmc-train/train/S03'
    video_path = {
        'c010': "{}/c010/vdo.avi".format(base),
        'c011': "{}/c011/vdo.avi".format(base),
        'c012': "{}/c012/vdo.avi".format(base),
        'c013': "{}/c013/vdo.avi".format(base),
        'c014': "{}/c014/vdo.avi".format(base),
        'c015': "{}/c015/vdo.avi".format(base),
    }
    frame_size = {
        'c010': [1920, 1080],
        'c011': [2560, 1920],
        'c012': [2560, 1920],
        'c013': [2560, 1920],
        'c014': [1920, 1080],
        'c015': [1920, 1080]
    }

    camerasList = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    pickle_names = ['c10', 'c11', 'c12', 'c13', 'c14', 'c15']

    # Load tracking detections
    dets = []
    for pickle_name in pickle_names:
        file = "tracks_" + pickle_name + ".pkl"
        with open(file, 'rb') as f:
            dets.append(pickle.load(f))
            f.close()

    detections, _, _, _, _ = format_pkl(all_pkl=dets, camerasL=camerasList, isGt=False, correctOffset=False,
                                        timestamp=timestamp, fps_r=fps_r)

    groundTruthdata, vid_paths = train_features.get_gt_info(gt_train_paths)
    nca = train_features.train(groundTruthdata, vid_paths)

    index_box_track = [0.25, 0.5, 0.60]
    count_cams = 1
    total_scores = []
    corrected_detections = detections

    for idCam in trange(len(detections) - 1):
        cam1 = camerasList[idCam]
        cam2 = camerasList[count_cams]
        count_cams += 1
        num_tracks1 = len(detections[idCam][cam1])
        num_tracks2 = len(detections[idCam + 1][cam2])
        for i in range(num_tracks1):
            track1 = detections[idCam][cam1][i]
            cropped_bboxes_cam1 = crop_image(track1, index_box_track)
            for j in range(num_tracks2):
                track2 = detections[idCam + 1][cam2][j]
                cropped_bboxes_cam2 = crop_image(track2, index_box_track)
                scores = []
                for n in range(len(cropped_bboxes_cam1)):
                    ft_vecCam1 = histogram_hue(cropped_bboxes_cam1[n])
                    ft_vecCam2 = histogram_hue(cropped_bboxes_cam2[n])
                    formPairs = [np.vstack((ft_vecCam1, ft_vecCam2))]
                    formPairs = np.array(formPairs)
                    score = nca.score_pairs(formPairs).mean()
                    scores.append(score)
                mean_score = np.mean(np.array(scores))
                total_scores.append(scores)
                if mean_score < 50:
                    track2['track_id'] = track1['track_id']
                    # Update the detections
                    corrected_detections[idCam][cam1][i] = track1
                    corrected_detections[idCam + 1][cam2][j] = track2

    cams_List, trackId_List, frames_List, boxes_List = reformat_predictions(corrected_detections)

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

            cv2.imshow('Video', im)
            out.write(im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vidCapture.release()
        out.release()
        cv2.destroyAllWindows()

    det_info = reformat_predictions(corrected_detections)
    compute_score(det_info)


task2()
