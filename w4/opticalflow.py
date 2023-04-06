import sys
from copy import deepcopy

import numpy as np
import cv2

sys.path.append("../")

#https://github.com/pathak22/pyflow.git
from pyflow import pyflow

#https://github.com/philferriere/tfoptflow
from tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from tfoptflow.tfoptflow.visualize import display_img_pairs_w_flows



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


def compute_epicflow(img, img_prev):
    # Following the guide of https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/pwcnet_eval_lg-6-2-multisteps-chairsthingsmix_flyingchairs.ipynb

    # Build a list of image pairs to process
    img_pairs = []
    img_pairs.append((img_prev, img))

    # Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
    gpu_devices = ['/device:CPU:0']  
    controller = '/device:CPU:0'

    ckpt_path = '../tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

    # Configure the model for inference, starting with the default options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller

    # We're running the PWC-Net-large model in quarter-resolution mode
    # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
    # of 64. Hence, we need to crop the predicted flows to their original size
    nn_opts['adapt_info'] = (1, 436, 1024, 2)

    # Instantiate the model in inference mode and display the model configuration
    nn = ModelPWCNet(mode='test', options=nn_opts)
    nn.print_config()

    pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1)
    return pred_labels