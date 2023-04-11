import numpy as np
import matplotlib.pyplot as plt

def calculate_msen(gt_flow, pred_flow):
    """
    Function to compute  the Mean Square Error in Non-occluded areas
    gt_flow: the ground thruth optical flow
    pred_flow: the predicted optical flow
    """
    # Get the mask of the valid points
    mask = gt_flow[:, :, 2] == 1

    # compute the error in du and dv
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    msen = np.mean(sqrt_error_masked)

    return msen


def calculate_pepn(gt_flow, pred_flow, th):
    """
    Function to compute the percentage of Erroneous Pixels in Non-occluded areas
    gt_flow: the ground thruth optical flow
    pred_flow: the predicted optical flow
    """

    # Get the mask of the valid points
    mask = gt_flow[:, :, 2] == 1

    # compute the error in du and dv
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    
    #error_plt = np.zeros(gt_flow.shape)
    #error_plt[mask,0] = sqrt_error[mask]
    #error_plt[error_plt < th] = 0
    #plt.imshow(error_plt)
    #plt.show()

    #plt.hist(sqrt_error_masked, bins=10, density=True)
    #plt.xlabel('Error')
    #plt.ylabel('Percentage')
    #plt.title('PEPN Histogram')
    #plt.show()

    return np.sum(sqrt_error_masked > th) / len(sqrt_error_masked)

def frameIOU(boxA, boxB):
    # For each prediction, compute its iou over all the boxes in that frame
    xleft1, yleft1, xright1, yright1 = np.split(boxA, 4, axis=0)
    xleft2, yleft2, xright2, yright2 = np.split(boxB, 4, axis=0)

    # Calculate the intersection in the bboxes
    xmin = np.maximum(xleft1, np.transpose(xleft2))
    ymin = np.maximum(yleft1, np.transpose(yleft2))
    xmax = np.minimum(xright1, np.transpose(xright2))
    ymax = np.minimum(yright1, np.transpose(yright2))
    w = np.maximum(xmax - xmin + 1.0, 0.0)
    h = np.maximum(ymax - ymin + 1.0, 0.0)
    intersection = w * h

    # Calculate the Union in the bboxes
    areaboxA = (xright1 - xleft1 + 1.0) * (yright1 - yleft1 + 1.0)
    areaboxB = (xright2 - xleft2 + 1.0) * (yright2 - yleft2 + 1.0)
    union = areaboxA + np.transpose(areaboxB) - intersection

    # Calculate IOU
    iou = intersection / union
    maxScore = max(iou)
    index = np.argmax(iou)

    # Return the IOU, maxscore, index
    return iou, maxScore, index