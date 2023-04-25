import os
import pickle
import cv2
import numpy as np


# from https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
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


def histogram_hue(image, splits=(3, 3), square_root=False):
    bins = 32
    x_splits, y_splits = splits
    x_len = int(image.shape[0] / x_splits)
    y_len = int(image.shape[1] / y_splits)

    histograms = []

    for i in range(x_splits):
        for j in range(y_splits):
            roi = image[i * x_len: (i + 1) * x_len, j * y_len: (j + 1) * y_len]
            roi_mask = None

            if len(roi.shape) == 3:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                roi_hist = \
                    np.histogram(roi_hsv[..., 0][roi_mask], bins=bins, density=True, range=(0, 180))[0]
                histograms.append(roi_hist)
            else:
                raise Exception("Image should have more channels")

    histograms = [np.sqrt(histogram) if square_root else histogram for histogram in histograms]
    return np.concatenate(histograms, axis=0)


# Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/blob/main/week5
def format_pkl(all_pkl, camerasL):
    allDetections = []
    for i, cam in enumerate(camerasL):
        data = []
        for j, id in enumerate(all_pkl[i]['id']):
            detections = []
            list_frames = all_pkl[i]['frame'][j]
            if len(np.where(np.array(list_frames) == -1)[0]) > 0:
                del list_frames[-1]
            for k, frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]
                detections.append({'frame': frame, 'box': boxes})
            data.append({'track_id': id, 'info': detections})

        allDetections.append({cam: data})

    return allDetections


def format_gt_pkl(all_pkl, camerasL):
    allDetections = []
    for i, cam in enumerate(camerasL):
        data = []
        for j, id in enumerate(all_pkl[i]['id']):
            detections = []
            list_frames = all_pkl[i]['frame'][j]
            if len(np.where(np.array(list_frames) == -1)[0]) > 0:
                del list_frames[-1]
            for k, frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]
                detections.append({'frame': frame, 'box': boxes})
            data.append({'track_id': id, 'info': detections})

        allDetections.append({cam: data})

    return allDetections


def import_gt_track(gt_path, outpath='gt_tracks.pkl', save=True):
    dict = {}
    ids = []
    frames = {}
    bboxes = {}

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(",")

            id = int(fields[1])
            frame = int(fields[0])
            bbox = [int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5])]

            if id in ids:
                frames[id].append(frame)
                bboxes[id].append(bbox)
            else:
                ids.append(id)
                frames[id] = [frame]
                bboxes[id] = [bbox]

    dict['id'] = ids
    dict['frame'] = list(frames.values())
    dict['box'] = list(bboxes.values())

    if save:
        outfile = open(outpath, 'wb')
        pickle.dump(dict, outfile)
        outfile.close()

    return dict


def get_gt_info(gt_paths):
    camerasL = []
    data = {'id': [], 'frame': [], 'box': [], 'cam': []}
    video_paths = {}

    for seq_path in gt_paths:
        for cam_path in os.listdir(seq_path):
            camerasL.append(cam_path)
            gt_file_path = os.path.join(seq_path, cam_path, 'gt', 'gt.txt')
            video_paths[cam_path] = os.path.join(seq_path, cam_path, 'vdo.avi')
            gt_data = import_gt_track(gt_file_path)

            data['id'].extend(gt_data['id'])
            data['frame'].extend(gt_data['frame'])
            data['box'].extend(gt_data['box'])
            data['cam'].extend([cam_path] * len(gt_data['id']))

    id, fr, bo, ca = zip(*sorted(zip(data['id'], data['frame'], data['box'], data['cam'])))
    data['id'], data['frame'], data['box'], data['cam'] = list(id), list(fr), list(bo), list(ca)

    return data, video_paths

def crop_image(track, index_box_track):
    croppedIms = []
    for i in range(len(index_box_track)):
        id = index_box_track[i]
        bbox = track['info'][int(id * len(track['info']))]['box']
        bbox = [int(p) for p in bbox]
        path = track['info'][int(id * len(track['info']))]['frame_path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropIm = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        croppedIms.append(cropIm)

    return croppedIms


def centroid(box):  # box [x,y,w,h]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    x_center = (box[0] + x2) / 2
    y_center = (box[1] + y2) / 2
    return x_center, y_center


def reformat_predictions(correct_pred):
    boxes_List = []
    frames_List = []
    trackId_List = []
    cams_List = []
    for i in range(len(correct_pred)):
        cam = list(correct_pred[i].keys())[0]
        for j in range(len(correct_pred[i][cam])):
            track_id = correct_pred[i][cam][j]['track_id']
            for k in range(len(correct_pred[i][cam][j]['info'])):
                frame = correct_pred[i][cam][j]['info'][k]['frame']
                boxes = correct_pred[i][cam][j]['info'][k]['box']
                cams_List.append(cam)
                trackId_List.append(track_id)
                frames_List.append(frame)
                boxes_List.append(boxes)

    frames_List, boxes_List, trackId_List, cams_List = zip(
        *sorted(zip(frames_List, boxes_List, trackId_List, cams_List)))
    return list(cams_List), list(trackId_List), list(frames_List), list(boxes_List)


def format_pkl(all_pkl, camerasL, isGt, correctOffset, timestamp, fps_r):
    framesS03 = 'aic19-track1-mtmc-train/train/S03'
    allDetections = []
    boxes_List = []
    frames_List = []
    trackId_List = []
    cams_List = []
    for i, cam in enumerate(camerasL):
        data = []
        for j, id in enumerate(all_pkl[i]['id']):
            detections = []
            list_frames = all_pkl[i]['frame'][j]
            if len(np.where(np.array(list_frames) == -1)[0]) > 0:
                del list_frames[-1]
            for k, frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]
                cams_List.append(cam)
                trackId_List.append(id)
                frames_List.append(frame)
                boxes_List.append(boxes)
                if correctOffset:
                    frame = int(frame * fps_r[cam] + timestamp[cam])
                if not isGt:
                    frame_path = "{}/frames/{}.jpg".format(os.path.join(framesS03, cam), str(frame).zfill(5))
                    detections.append({'frame': frame, 'frame_path': frame_path, 'box': boxes})
                else:
                    detections.append({'frame': frame, 'box': boxes})
            data.append({'track_id': id, 'info': detections})

        allDetections.append({cam: data})
    frames_List, boxes_List, trackId_List, cams_List = zip(
        *sorted(zip(frames_List, boxes_List, trackId_List, cams_List)))

    return allDetections, list(cams_List), list(trackId_List), list(frames_List), list(boxes_List)
