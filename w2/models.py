import pickle

import cv2
import numpy as np
from tqdm import trange


# Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/tree/main/week2
def findBBOX(mask):
    minH = 50
    maxH = 1080 / 2
    minW = 100
    maxW = 1920 / 2

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if minW < w < maxW and minH < h < maxH:
            if 0.2 < w / h < 10:
                box.append([x, y, x + w, y + h])

    return box


class GaussianModel:
    def __init__(self, path, colorSpace):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.colorSpace = colorSpace
        self.mean = []
        self.std = []
        self.channels = 3

        # Change channels according to colorspace
        if self.colorSpace == "gray":
            self.color_transform = cv2.COLOR_BGR2GRAY
            self.channels = 1
        elif self.colorSpace == "hsv":
            self.color_transform = cv2.COLOR_BGR2HSV
        elif self.colorSpace == "rgb":
            self.color_transform = cv2.COLOR_BGR2RGB

    # Function to find length
    def find_length(self):
        return self.num_frames

    def calculate_mean_std(self):
        # Use 25 percent of the frames
        len_25 = int(self.num_frames * 0.25)
        frames = np.zeros((len_25, self.height, self.width, self.channels), dtype=np.uint8)

        # Loop over the frames
        for i in trange(len_25, desc='GaussianModelling background'):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            image = self.cap.read()[1]
            image = cv2.cvtColor(image, self.color_transform)
            image = image.reshape(image.shape[0], image.shape[1], self.channels)
            frames[i] = image

        # Calculate mean and std
        self.mean = np.mean(frames, axis=0)
        self.std = np.std(frames, axis=0)

        # Save the mean and std
        with open('mean.pkl', 'wb') as handle:
            pickle.dump(self.mean, handle)
        with open('std.pkl', 'wb') as handle:
            pickle.dump(self.std, handle)

    # Foreground extraction task 1
    def model_foreground(self, alpha):
        # From 25-100
        start = int(self.num_frames * 0.25)
        end = int(self.num_frames)

        # Initialize lists
        predictedBBOX = []
        predictedFrames = []
        count = 0

        # Loop over the frames
        for i in trange(start, end, desc='Foreground extraction'):
            # Read the image
            image = self.cap.read()[1]
            image = cv2.cvtColor(image, self.color_transform)
            image = image.reshape(image.shape[0], image.shape[1], self.channels)

            # Calculate mask by criterion
            mask = abs(image - self.mean) >= (alpha * self.std+2)
            mask = mask * 1.0
            mask = mask.reshape(mask.shape[0], mask.shape[1], self.channels)

            if self.channels > 1:
                mask = cv2.cvtColor(mask, self.cv2.COLOR_BGR2GRAY)

            # Denoise mask
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

            # Find the box
            bboxFrame = findBBOX(closing)
            predictedBBOX.append(bboxFrame)
            predictedFrames.append(i)

        # Make the format same as last week
        predictionInfo = []
        num_boxes = 0

        # Loop over the boxes
        for i in range(len(predictedBBOX)):
            boxes = predictedBBOX[i]
            predictionInfo.append({"frame": predictedFrames[i], "bbox": np.array(boxes)})
            num_boxes = num_boxes + len(boxes)

        # Return
        return predictionInfo, num_boxes
