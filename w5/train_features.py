# Import required packages
from matplotlib import pyplot as plt
from metric_learn import NCA
import random
from utils import *
from tqdm import trange


# http://contrib.scikit-learn.org/metric-learn/generated/metric_learn.NCA.html
def train(groundTruth, video_paths, display=False):
    """
    Trains a NCA Model on the provided data and returns the trained model.

    Args:
        groundTruth (numpy array): Ground truth data consisting of frame numbers, bounding boxes, camera numbers,
                              and track IDs.
        video_paths (list): List of file paths for the video sequences used in the ground truth data.
        display (bool, optional): If True, displays pairs of images with the same IDs across tracks. Defaults to False.

    Returns:
        Trained NCA model.

    """
    # Set number of images to get from each track
    num_track = 30

    # Extract features and labels from data
    new_features = []
    new_labels = []
    uniq_tracks = np.unique(groundTruth['id'])
    for id_tr in trange(len(uniq_tracks), desc="get gt data"):
        track_id = uniq_tracks[id_tr]

        # Get all the indices that have the same tracking number
        indices = [i for i, x in enumerate(groundTruth['id']) if x == track_id]

        # Extract frames, bboxes, and cameras from the indices
        frames = groundTruth['frame']
        frames = [frames[i] for i in indices]
        bboxes = groundTruth['box']
        bboxes = [bboxes[i] for i in indices]
        cameras = groundTruth['cam']
        cameras = [cameras[i] for i in indices]

        cam_frames = []
        for i in range(len(frames)):
            cam_frames.append([cameras[i]] * len(frames[i]))

        # Flatten lists
        frames = [item for sublist in frames for item in sublist]
        bboxes = [item for sublist in bboxes for item in sublist]
        cam_frames = [item for sublist in cam_frames for item in sublist]

        # If more than num_track frames, select num_track frames randomly
        if len(frames) > num_track:
            indices = random.sample(range(len(frames)), num_track)
            frames = [frames[i] for i in indices]
            bboxes = [bboxes[i] for i in indices]
            cam_frames = [cam_frames[i] for i in indices]

        # Extract features for each bbox image and append to new_features list
        for i in range(len(frames)):
            # Get bbox from image
            videoPath = video_paths[cam_frames[i]]
            cap = cv2.VideoCapture(videoPath)
            cap.set(1, frames[i] - 1)
            ret, vid_frame = cap.read()
            boundingBox = bboxes[i]
            bbox_img = cv2.cvtColor(vid_frame[boundingBox[1]:boundingBox[1] + boundingBox[3], boundingBox[0]:boundingBox[0] + boundingBox[2], :], cv2.COLOR_BGR2RGB)

            # Get histogram
            histogram_features = histogram_hue(bbox_img)

            # Append the features
            new_features.append(histogram_features)
            new_labels.append(track_id)
            cap.release()

    # Convert new_features and new_labels to numpy arrays
    X = np.array(new_features)
    Y = np.array(new_labels)

    # Fit NCA to the data
    nca = NCA(init='pca', n_components=400, max_iter=1500, verbose=True)
    nca.fit(X, Y)

    if display:
        # Display pairs of images with same IDs across tracks
        unique_ids = np.unique(groundTruth['id'])
        for i in range(len(unique_ids)):
            id_ = unique_ids[i]
            indices = np.where(groundTruth['id'] == id_)[0]
            if len(indices) > 1:
                # Select two random frames with the same ID from different tracks
                rand_indices = np.random.choice(indices, size=2, replace=False)
                frame1 = groundTruth['frame'][rand_indices[0]][0]
                frame2 = groundTruth['frame'][rand_indices[1]][0]
                cam1 = groundTruth['cam'][rand_indices[0]]
                cam2 = groundTruth['cam'][rand_indices[1]]

                # Load the frames and display them side by side
                videoPath1 = video_paths[cam1]
                cap1 = cv2.VideoCapture(videoPath1)
                cap1.set(1, frame1 - 1)
                ret1, vid_frame1 = cap1.read()
                vid_frame1 = cv2.cvtColor(vid_frame1, cv2.COLOR_BGR2RGB)
                vid_frame1 = cv2.resize(vid_frame1, (960, 960))

                # Load the frames and display them side by side
                videoPath2 = video_paths[cam2]
                cap2 = cv2.VideoCapture(videoPath2)
                cap2.set(1, frame2 - 1)
                ret2, vid_frame2 = cap2.read()
                vid_frame2 = cv2.cvtColor(vid_frame2, cv2.COLOR_BGR2RGB)
                vid_frame2 = cv2.resize(vid_frame2, (960, 960))

                # Concatenate the frames
                concat_frames = np.concatenate((vid_frame1, vid_frame2), axis=1)
                plt.imshow(concat_frames)
                plt.show()

                # Close the video
                cap1.release()
                cap2.release()

    # Return classifier
    return nca
