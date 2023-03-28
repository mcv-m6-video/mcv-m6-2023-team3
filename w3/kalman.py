import os
import pickle
from sort import Sort


import numpy as np
import cv2


from read_data import VideoData, parse_annotations, read_frame_boxes

AICITY_DATA_PATH = 'AICity_data/train/S03/c010'

# Read gt file
ANOTATIONS_PATH = 'ai_challenge_s03_c010-full_annotation.xml'
VIDEO_PATH = os.path.join(AICITY_DATA_PATH,"vdo.avi")
GENERATE_VIDEO = False

def load_frame_boxes(path):
    # Get the bboxes
    frame_bboxes = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                frame_bboxes.append(pickle.load(openfile))
            except EOFError:
                break
    frame_bboxes = frame_bboxes[0]
    return frame_bboxes

def centroid(box): # box [x,y,w,h]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    x_center = (box[0] + x2) / 2
    y_center = (box[1] + y2) / 2
    return x_center, y_center



def main():

    print("Load video")
    # Load the boxes from fasterRCNN
    all_tracks = []
    predict_boxes = []
    kalman_box_id = []
    
    frame_bboxes = load_frame_boxes("w3/boxesScores1.pkl")
    gt_info = parse_annotations(path=ANOTATIONS_PATH, isGT=True)
    tracker = Sort()

    for n_frame, frame_bbox in enumerate(frame_bboxes):
        dets = read_frame_boxes(frame_bbox)
        tracks = tracker.update(dets)
        all_tracks.append(tracks)

        # Compute the centroids for ground thruth
        gt_list = [box["bbox"] for box in gt_info if box["frame"] == n_frame]
        gt_centroids = np.array([centroid(box) for box in gt_list[0]])

        # Compute the centroids for Kalman filter predicitons
        boxes_detected = tracks[:, 0:4] # take only the box coordinate from the predicition
        predict_boxes.append(boxes_detected)
        kalman_box_id.append(tracks[:, 4]) # take the id
        kalman_centroids = np.array([centroid([item[0], item[1], item[2] - item[0], item[3] - item[1]]) for item in boxes_detected])
        print(kalman_centroids)
        
        distances = []
        for centroid_gt in gt_centroids:
            distance = []
            for k_centroid in kalman_centroids:
                d = np.linalg.norm(centroid_gt - k_centroid) # euclidian distance
                distance.append(d) 
                print("The distance is:", d)
            distances.append(distance)
    
    if GENERATE_VIDEO:
        colors = []
        max_id = max([max(list(k)) for k in kalman_box_id])
        for i in range(int(max_id)+1):
            color = list(np.random.choice(range(256), size=3))
            colors.append(color)

        video_capturer = cv2.VideoCapture(VIDEO_PATH)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('task22.avi', fourcc, 10.0, (1920, 1080))
        
        
        for i in range(len(frame_bboxes)):
            video_capturer.set(cv2.CAP_PROP_POS_FRAMES, i)
            im = video_capturer.read()[1]
            for id in range(len(kalman_box_id[i])):
                id_index = int(kalman_box_id[i][id])
                box = predict_boxes[i][id]
                color = colors[id_index]
                cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]-box[0]), int(box[3]-box[1])),
                              (int(color[0]), int(color[1]), int(color[2])), 2)
            video_writer.write(im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capturer.release()
        video_writer.release()
        cv2.destroyAllWindows()

main()