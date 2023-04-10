import sys

import numpy as np
import cv2

sys.path.append("../")

#https://github.com/pathak22/pyflow.git
from pyflow import pyflow


def compute_pyflow(img, img_prev, color=1):
    """
        color: 0  for RGB,  1 for GRAY 
    """
    channels = 1 if color==1 else 3
    img_prev_norm = np.reshape(img_prev.astype(float) / 255.,[img_prev.shape[0], img_prev.shape[1],channels])
    img_norm = np.reshape(img.astype(float) / 255.,[img.shape[0], img.shape[1],channels])

    alpha = 0.012
    ratio = 0.75
    min_width = 20
    num_outer_fp = 7
    num_inner_fp = 1
    n_iterations = 30

    u, v, im2W = pyflow.coarse2fine_flow(img_prev_norm, img_norm, alpha, ratio, min_width, num_outer_fp,
                                            num_inner_fp, n_iterations, color)

    flow = np.dstack((u, v))
    return flow


def compute_lukas_kanade(img, img_prev):
    """
        Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
    """
    height, width = img_prev.shape
    
    # Build the vector of 2D points for which the flow needs to be found; 
    x, y = np.meshgrid(range(width), range(height))
    img_points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1, dtype=np.float32).reshape((-1, 1, 2))

    params = {
        "winSize":(10, 10),
        "maxLevel":3,
        "criteria":(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    img_points_next, status, err = cv2.calcOpticalFlowPyrLK(img_prev, img, img_points, None, **params)

    img_points = img_points.reshape((height, width, 2))
    img_points_next = img_points_next.reshape((height, width, 2))
    status = status.reshape((height, width))

    flow = img_points_next - img_points
    flow[status == 0] = 0
    return flow
