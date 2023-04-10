import numpy as np


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

    return np.sum(sqrt_error_masked > th) / len(sqrt_error_masked)
