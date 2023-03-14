import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from PIL import Image

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