import numpy as np
import xmltodict
import cv2


def parse_annotations(path, isGT=False, startFrame=0):
    # Open the file
    with open(path) as file:
        tracks = xmltodict.parse(file.read())['annotations']['track']

    # Define list of frames and bounding boxes and ground truth
    frames = []
    BBOX = []
    groundTruth = []

    # Loop over the tracks
    for track in tracks:
        id = track['@id']
        label = track['@label']
        boxes = track['box']
        for box in boxes:
            if label == 'car':
                parked = box['attribute']['#text'].lower() == 'true'
            else:
                parked = None
            gt_list = [int(box['@frame']), int(id), label, float(box['@xtl']), float(box['@ytl']),
                       float(box['@xbr']), float(box['@ybr']), float(-1), parked]
            groundTruth.append(gt_list)

    for gt in groundTruth:
        # Extract Frames and bounding boxes
        if gt[2] == "car":
            frame = [gt[0], gt[1]]
            if frame[0] >= startFrame:
                bbox = [gt[3], gt[4], gt[5], gt[6]]

                # Add the frames and bounding boxes
                frames.append(frame)
                BBOX.append(bbox)

    # Sort frames
    sortedFrames, sortedBbox = zip(*sorted(zip(frames, BBOX)))

    # Define lists
    bbox = []
    groundTruthComplete = []

    # Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/tree/main/week1
    # Loop over the sorted Boxes
    for i in range(len(sortedBbox)):
        if i == 0:
            bbox.append(sortedBbox[i])

        else:
            if sortedFrames[i] == sortedFrames[i - 1]:
                bbox.append(sortedBbox[i])
            else:
                if isGT:
                    groundTruthComplete.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "already_detected": [False] * len(bbox)})
                else:
                    groundTruthComplete.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox)})
                bbox = [sortedBbox[i]]

            if i + 1 == len(sortedBbox):
                if isGT:
                    groundTruthComplete.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "already_detected": [False] * len(bbox)})
                else:
                    groundTruthComplete.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox)})

    return groundTruthComplete, groundTruth


def getPredictions(path, isGT=False):
    # Open file
    with open(path) as f:
        lines = f.readlines()

    # Define the list of frames, bounding boxes and confidence scores
    frames = []
    BBOX = []
    confidence_scores = []
    predictions = []

    # Loop over the tracks
    for line in lines:
        track = line.split(',')
        pred_list = [int(track[0]) - 1, track[1], 'car', float(track[2]), float(track[3]),
                     float(track[2]) + float(track[4]),
                     float(track[3]) + float(track[5]), float(track[6])]
        predictions.append(pred_list)

    # Loop over the predictions
    for prediction in predictions:
        # Grab the frames, bounding box and confidence for each prediction
        frame = prediction[0]
        bbox = [prediction[3], prediction[4], prediction[5], prediction[6]]
        confidence = prediction[7]

        # Append each variable respectively
        frames.append(frame)
        BBOX.append(bbox)
        confidence_scores.append(confidence)

    # Sort frames and corresponding bounding boxes and confidence scores
    sortedFrames, sortedBbox, sortedScore = zip(*sorted(zip(frames, BBOX, confidence_scores)))

    # Define list of bounding boxes, score
    boundingBox = []
    score = []
    detectedInfo = []

    # Loop over the sorted boxes
    for i in range(len(sortedBbox)):
        # If first frame add bounding box
        if i == 0:
            boundingBox.append(sortedBbox[i])
            score.append(sortedScore[i])

        else:
            if sortedFrames[i] == sortedFrames[i - 1]:
                boundingBox.append(sortedBbox[i])
                score.append(sortedScore[i])
            else:
                if isGT:
                    # Parameter to check if box is already detected for the frame
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score),
                         "already_detected": [False] * len(boundingBox)})
                else:
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score)})
                boundingBox = []
                score = []
                boundingBox.append(sortedBbox[i])
                score.append(sortedScore[i])

            if i + 1 == len(sortedBbox):
                if isGT:
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score),
                         "already_detected": [False] * len(boundingBox)})
                else:
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score)})

    # Return
    return detectedInfo


class VideoData:
    def __init__(self, video_path, color="gray"):
        self.video_data = cv2.VideoCapture(video_path)
        self.width = int(self.video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        self.channels = 3
        if color == "gray":
            self.color_transform = cv2.COLOR_BGR2GRAY
            self.channels = 1
        elif self.colorSpace == "hsv":
            self.color_transform = cv2.COLOR_BGR2HSV
        else:
            self.color_transform = cv2.COLOR_BGR2RGB

    def get_number_frames(self):
        return self.num_frames

    def conver_slice_to_grayscale(self, start_frame, last_frame):
        # frames = np.zeros((self.width, self.height, self.num_frames))
        frames = []
        for i in range(start_frame, last_frame, 1):
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, i)
            image = self.video_data.read()[1]
            image = cv2.cvtColor(image, self.color_transform)
            # print(np.shape(image))
            image = image.reshape(image.shape[0], image.shape[1], self.channels)
            frames.append(image)
        frames = np.array(frames)
        print(np.shape(frames))
        return frames

    def convert_frame_by_idx(self, frame_idx):
        self.video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        image = self.video_data.read()[1]
        image = cv2.cvtColor(image, self.color_transform)
        # print(np.shape(image))
        frame = image.reshape(image.shape[0], image.shape[1], self.channels)
        return frame


def read_frame_boxes(frame_box):
    frame_boxes = []
    for box in frame_box:
        # convert data to x1,y1,x2,y2
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[0][2] + x1
        y2 = box[0][3] + y1
        dets = np.array([x1, y1, x2, y2, box[1]])
        frame_boxes.append(dets)
    return np.array(frame_boxes)

def load_bb(path):
    """
    Loads the bounding boxes from a path.
    They list the ground truths of MTMC tracking in the MOTChallenge format 
    [frame, ID, left, top, width, height, 1, -1, -1, -1].
    Only frame, left, top, width and height are used.
    """
    bbs = []
    with open(path, 'r') as f:
        for line in f:
            # Split the line by commas
            bb = line.strip().split(',')
            # Extract values 0, 3, 4, 5, and 6
            bb_info = [float(bb[i]) for i in range(2, 6)]
            xtl = bb_info[0]
            ytl = bb_info[1]
            xbr = bb_info[0]+bb_info[3]
            ybr = bb_info[1]+bb_info[2]
            bb_info_new = [xtl,ytl,xbr,ybr]
            values = [int(bb[0]),int(bb[1])] + bb_info_new + [-1,int(bb[6])]
            bbs.append(values)
    return bbs