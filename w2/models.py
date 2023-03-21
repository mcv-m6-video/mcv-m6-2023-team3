import pickle

import cv2
import numpy as np
from tqdm import trange
import os
import imageio
import matplotlib.pyplot as plt

# Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/tree/main/week2
def findBBOX(mask):
    minH = 50
    maxH = 1080 / 2
    minW = 100
    maxW = 1920 / 2

    if cv2.__version__ == "3.4.2":
        _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
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
        self.mask_previous = None

        # Change channels according to colorspace
        if self.colorSpace == "gray":
            self.color_transform = cv2.COLOR_BGR2GRAY
            self.channels = 1
        elif self.colorSpace == "hsv":
            self.color_transform = cv2.COLOR_BGR2HSV
        elif self.colorSpace == "hs":
            self.color_transform = cv2.COLOR_BGR2HSV
            self.channels = 2
        elif self.colorSpace == "h":
            self.color_transform = cv2.COLOR_BGR2HSV
            self.channels = 1
        elif self.colorSpace == "rgb":
            self.color_transform = cv2.COLOR_BGR2RGB
        elif self.colorSpace == "yuv":
            self.color_transform = cv2.COLOR_BGR2Luv
            self.channels = 2
        elif self.colorSpace == "xyz":
            self.color_transform = cv2.COLOR_BGR2XYZ
            self.channels = 2

    # Function to find length
    def find_length(self):
        return self.num_frames

    def get_video_rec_25(self):

        start = int(self.num_frames * 0.03)
        len_25 = int(self.num_frames * 0.15)
        frames = np.zeros((len_25, self.height, self.width, self.channels), dtype=np.uint8)
        save_dir = './frame'
        gif_dir = 'gif.gif'
        point = (400,600)
        # Loop over the frames
        with imageio.get_writer(gif_dir, mode='I') as writer:
            for i in trange(start, len_25, desc='Video rect'):
                if (i - start) % 3 == 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    image = self.cap.read()[1]
                    image = cv2.cvtColor(image, self.color_transform)
                    if self.colorSpace != "gray":
                        image = image[:,:,:self.channels]
                    image = image.reshape(image.shape[0], image.shape[1], self.channels)
                    frames[i] = image
                    x1, y1 = point[0]-5, point[1]-5   # top-left corner
                    x2, y2 = point[0]+5, point[1]+5   # bottom-right corner
                    
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(save_dir, str(i) + '.png'), image)
                    image = imageio.imread(os.path.join(save_dir, str(i) + '.png'))
                    writer.append_data(image)

    def get_video_rec_25_plot(self):
        # Use 25 percent of the frames
        start = int(self.num_frames * 0.03)
        len_25 = int(self.num_frames * 0.15)
        frames = np.zeros((len_25-start, self.height, self.width, self.channels), dtype=np.uint8)
        save_dir = './frame'
        gif_dir = 'gif.gif'
        point = (400,600)
        plot_frames = []
        plot_mean = []
        plot_std = []
        # Loop over the frames
        with imageio.get_writer(gif_dir, mode='I') as writer:
            for i in trange(start, len_25, desc='Video rect'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                image = self.cap.read()[1]
                image = cv2.cvtColor(image, self.color_transform)
                if self.colorSpace != "gray":
                    image = image[:,:,:self.channels]
                image = image.reshape(image.shape[0], image.shape[1], self.channels)
                frames[i-start] = image

                plot_frames.append(i)
                

                plot_mean = np.hstack((plot_mean, np.mean(frames[:i-start+1,point[0],point[1]], axis=0)))      
                plot_std  = np.hstack((plot_std, np.std(frames[:i-start+1,point[0],point[1]], axis=0)))
                if (i - start) % 3 == 0:
                    fig, ax = plt.subplots()
                    
                    ax.plot(plot_frames, plot_mean, linewidth=0.5, label='mean')
                    plt.fill(np.append(plot_frames, plot_frames[::-1]), np.append(plot_mean + plot_std, (plot_mean - plot_std)[::-1]), 'powderblue',
                                    label='std')
                    ax.plot(plot_frames, frames[:i-start+1,point[0],point[1]], linewidth=0.5, label='gray scale pixel value')

                    ax.set_xlabel('Frames')
                    ax.set_ylabel('')
                    ax.set_title('')

                    ax.set_ylim([150, 190])
                    ax.set_xlim([start, len_25])
                    plt.legend(loc='upper center')
                    plt.savefig(os.path.join(save_dir, str(i) + '.png'))
                    plt.close()
                    image = imageio.imread(os.path.join(save_dir, str(i) + '.png'))
                    writer.append_data(image)

    def reduce_channels(self, image):
        if self.colorSpace == "hs":
            return image[:,:,[0,1]]
        elif self.colorSpace == "h":
            return image[:,:,0]
        elif self.colorSpace == "yuv":
            return image[:,:,[1,2]]
        elif self.colorSpace == "xyz":
            return image[:,:,[0,2]]
        else:
            return image
    def calculate_mean_std(self):
        # Use 25 percent of the frames
        len_25 = int(self.num_frames * 0.25)
        frames = np.zeros((len_25, self.height, self.width, self.channels), dtype=np.uint8)

        # Loop over the frames
        for i in trange(len_25, desc='GaussianModelling background'):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            image = self.cap.read()[1]
            image = cv2.cvtColor(image, self.color_transform)
            image = self.reduce_channels(image)
            image = image.reshape(image.shape[0], image.shape[1], self.channels)
            frames[i] = image

        # Calculate mean and std
        self.mean = np.mean(frames, axis=0)

        t, r, w, chs = frames.shape
        self.std = np.zeros((r, w, chs))
        for ch in range(chs):
            self.std[:, :, ch] = np.std(frames[:, :, :, ch], axis=0)

        # Save the mean and std
        with open('mean.pkl', 'wb') as handle:
            pickle.dump(self.mean, handle)
        with open('std.pkl', 'wb') as handle:
            pickle.dump(self.std, handle)

    # Foreground extraction task 1
    def model_foreground(self, alpha, gt):
        # From 25-100
        start = int(self.num_frames * 0.25)
        end = int(self.num_frames)

        # Initialize lists
        predictedBBOX = []
        predictedFrames = []
        count = 0
        with imageio.get_writer("alpha10.gif", mode='I') as writer:
            # Loop over the frames
            for i in trange(start, end, desc='Foreground extraction'):
                # Read the image
                image = self.cap.read()[1]
                image = cv2.cvtColor(image, self.color_transform)
                image = image.reshape(image.shape[0], image.shape[1], self.channels)

                # Calculate mask by criterion
                mask = abs(image - self.mean) >= (alpha * self.std + 2)
                mask = np.logical_or.reduce(mask, axis=2)
                mask = mask * 1.0
                mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

                # Denoise mask
                kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

                opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

                # Find the box
                bboxFrame = findBBOX(closing)
                predictedBBOX.append(bboxFrame)
                predictedFrames.append(i)
        
                gtBoxes = gt[i-start]['bbox']
                mRGB = np.zeros((closing.shape[0], closing.shape[1], 3))
                mRGB[:, :, 0] = closing*255
                mRGB[:, :, 1] = closing*255
                mRGB[:, :, 2] = closing*255

                for k in range(len(gtBoxes)):
                    gbox = gtBoxes[k]
                    if gbox is not None:
                        cv2.rectangle(mRGB, (int(gbox[0]), int(gbox[1])), (int(gbox[2]), int(gbox[3])), (0, 0, 255), 2)
                for b in bboxFrame:
                    cv2.rectangle(mRGB, (b[0], b[1]), (b[2], b[3]), (100, 255, 0), 2)

                cv2.imwrite(os.path.join("output", str(i) + '.png'), mRGB)
                image = imageio.imread(os.path.join("output", str(i) + '.png'))
                writer.append_data(image)
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

    def model_foreground_Adaptive(self, alpha, rho):
        # From 25-100
        start = int(self.num_frames * 0.25)
        end = int(self.num_frames)

        # Initialize lists
        predictedBBOX = []
        predictedFrames = []
        count = 0

        for i in trange(start, end, desc='Adaptive Foreground extraction'):
            # Read the image
            image = self.cap.read()[1]
            image = cv2.cvtColor(image, self.color_transform)
            image = self.reduce_channels(image)
            image = image.reshape(image.shape[0], image.shape[1], self.channels)

            if self.mask_previous is not None:
                self.mean = (1 - rho) * self.mean
                self.std = np.sqrt(rho * (image * (1 - self.mask_previous) - self.mean) ** 2 + (1 - rho) * self.std ** 2)

            # Calculate mask by criterion
            mask = abs(image - self.mean) >= (alpha * self.std + 2)
            mask = np.logical_or.reduce(mask, axis=2)
            mask = mask * 1.0
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            self.mask_previous = mask

            # Denoise mask
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

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


class AdaptativeBackEstimator:
    def __init__(self, roi, size, color_format="grayscale") -> None:
        self.color_format = color_format
        self.roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
        self.height = size[0]
        self.width = size[1]
        self.channels = size[2]

    def train(self, video_frames):
        len_video_frames = len(video_frames)

        mean_img = np.zeros([self.height, self.width, self.channels], dtype=np.float32)
        std_img = np.zeros([self.height, self.width, self.channels], dtype=np.float32)

        for frame in video_frames:
            mean_img = mean_img + frame / len_video_frames

        for frame in video_frames:
            std_img = std_img + ((mean_img - frame) ** 2) / (len_video_frames - 1)

        std_img = np.sqrt(std_img)

        self.mean = mean_img
        self.std = std_img
        return mean_img, std_img

    def evaluate(self, frame, rho=0.02, alpha=4, ):
        # Update background model
        foreground_gaussian_model = (np.abs(frame - self.mean) >= alpha * (self.std + 2))
        foreground_gaussian_model = foreground_gaussian_model * self.roi
        background = ~foreground_gaussian_model

        self.mean[background] = rho * frame[background] + (1 - rho) * self.mean[background]
        self.std[background] = np.sqrt(
            rho * (frame[background] - self.mean[background]) ** 2 + (1 - rho) * self.std[background] ** 2)
        filtered_foreground_gaussian_model = self.morphological_filtering(foreground_gaussian_model.astype(np.uint8))

        detection = findBBOX(filtered_foreground_gaussian_model)

        # Return
        return detection, filtered_foreground_gaussian_model

    # https://github.com/mcv-m6-video/mcv-m6-2022-team3/blob/main/week2/morphology_utils.py
    def morphological_filtering(self, mask):
        # 1. Remove noise
        mask = cv2.medianBlur(mask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 2. Connect regions and remove shadows
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 3. Fill convex hull of connected components
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

        mask2 = np.zeros_like(mask)
        for i in range(len(hull_list)):
            mask2 = cv2.drawContours(mask2, hull_list, i, color=1, thickness=cv2.FILLED)
        mask = mask2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
