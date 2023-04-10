import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import flow_vis

matplotlib.use('TkAgg')


def plot_ap_miou(study_noise_pos, title="AP vs MIOU", xlabel = ""):
    
    # Separate x and y values into separate lists
    distance = [x for x, _ in study_noise_pos]
    AP_values = [y[1] for _, y in study_noise_pos]
    MIOU_values = [y[0] for _, y in study_noise_pos]

    # Create a scatter plot
    plt.plot(distance, AP_values, label='AP')
    plt.plot(distance, MIOU_values, label='MIOU')

    # Add axis labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(loc='upper right')
    # Show the plot
    plt.show()



def plot_bb(img, gt, predict):
    image_copy = img.copy()

    # Create a subplot and display the image
    fig, ax = plt.subplots()
    ax.imshow(image_copy)

    # Add the bounding boxes to the plot
    for bbox in predict:
        _, left, top, width, height = bbox
        rect = Rectangle((left, top), width, height, fill=False, edgecolor='blue')
        ax.add_patch(rect)
    for bbox in gt:
        _, left, top, width, height = bbox
        rect = Rectangle((left, top), width, height, fill=False, edgecolor='red')
        ax.add_patch(rect)
    # Show the plot
    plt.show()

def plot_optical_flow(img, flow):
    # Set stride to skip every n pixels
    n = 10
    stride = (slice(None, None, n), slice(None, None, n))

    # Create meshgrid of x and y coordinates
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(0, w, n), np.arange(0, h, n))
    bright_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    # Select a subset of the optical flow data
    u, v = flow[stride][:,:,0], flow[stride][:,:,1]

    # Normalize the optical flow vectors
    max_flow = np.max(np.abs(flow))
    u_norm, v_norm = u / max_flow, v / max_flow
    cmap = plt.cm.jet
    mag = np.sqrt(u**2 + v**2)
    norm = plt.Normalize(vmin=0, vmax=np.max(mag))
    # Create quiver plot
    plt.imshow(bright_img)
    plt.quiver(x, y, u_norm, -v_norm, mag, cmap=cmap, norm=norm, scale=10, width=0.001)

    # Display the image and the quiver plot
    plt.show()

import os
def plot_histogram(title, img, gt_flow, pred_flow, save_path='.'):
    mask = gt_flow[:, :, 2] == 1

    # compute the error in du and dv
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    mean = np.mean(sqrt_error_masked)
    plt.figure()
    plt.title(title)
    plt.hist(sqrt_error_masked, 25, density=True, color="blue")
    plt.axvline(mean, color='g', linestyle='dashed', linewidth=1, label=f'MSEN {round(mean, 1)}')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'histogram'+img))
    plt.close()

def magnitudeOP(img, flow_img):
    flow_color = flow_vis.flow_to_color(flow_img[:, :, :2], convert_to_bgr=False)
    plt.imshow(flow_color)
    plt.imshow(img, alpha=0.2, cmap='gray')
    plt.title('Magnitude OF')
    plt.xticks([])
    plt.yticks([])
    plt.show()



def plot_3D(blockSize, searchAreas, msen, xlabel, ylabel, zlabel):
    blockSize = np.array(blockSize)
    searchAreas = np.array(searchAreas)
    searchAreas, blockSize = np.meshgrid(searchAreas, blockSize)

    msen = np.array(msen)
    # create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # create a surface plot with blockSize and searchAreas as x and y, and msen as z
    surf = ax.plot_surface(blockSize, searchAreas, msen, cmap='hot')

    # set the labels for the axes and colorbar
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # show the plot
    plt.show()

def magnitudeOP(img, flow_img):
    flow_color = flow_vis.flow_to_color(flow_img[:, :, :2], convert_to_bgr=False)
    plt.imshow(flow_color)
    plt.imshow(img, alpha=0.2, cmap='gray')
    plt.title('Magnitude OF')
    plt.xticks([])
    plt.yticks([])
    plt.show()
