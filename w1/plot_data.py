import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import cv2

matplotlib.use('TkAgg')


def plot_ap_miou(study_noise_pos, title="AP vs MIOU", xlabel = ""):
    
    # Separate x and y values into separate lists
    distance = [x for x, _ in study_noise_pos]
    AP_values = [y[1] for _, y in study_noise_pos] #TODO
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