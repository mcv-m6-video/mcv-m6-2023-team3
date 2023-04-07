import cv2
import numpy as np
from plot_data import plot_optical_flow

import sys
sys.path.append("./w4/tfoptflow/tfoptflow")
sys.path.append("./w4/OpticalFlowToolkit/lib")

from opticalflow import compute_pyflow, compute_lukas_kanade, compute_epicflow
from OpticalFlowToolkit.lib import flowlib
from compute_metric import calculate_msen, calculate_pepn

ERROR_THRESH = 3


def task1_2(method="pyflow"):

    img_prev = cv2.imread('./data_w1/img/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('./data_w1/img/000045_11.png', cv2.IMREAD_GRAYSCALE)
    gt_flow = flowlib.read_flow('./data_w1/flow_noc/000045_10.png')

    if method == 'pyflow':
        pred_flow = compute_pyflow(img, img_prev)

    elif method == 'kanade':
        pred_flow = compute_lukas_kanade(img, img_prev)
    elif method == 'epic':
        pred_flow = compute_epicflow(img, img_prev)
    
    plot_optical_flow(img, pred_flow)
    msen = calculate_msen(gt_flow, pred_flow)
    print("The msen for the img: " + str(img) + " is: " + str(msen))
    pepn = calculate_pepn(gt_flow, pred_flow, ERROR_THRESH)
    print("The pepn for the img: " + str(img) + " is: " + str(pepn))

task1_2()