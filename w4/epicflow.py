from copy import deepcopy

import numpy as np


#https://github.com/philferriere/tfoptflow
from tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from tfoptflow.tfoptflow.visualize import display_img_pairs_w_flows



def compute_epicflow(img, img_prev):
    # Following the guide of https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/pwcnet_eval_lg-6-2-multisteps-chairsthingsmix_flyingchairs.ipynb

    # Build a list of image pairs to process
    img_pairs = []
    img_pairs.append((img_prev, img))

    # Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
    gpu_devices = ['/device:CPU:0']  
    controller = '/device:CPU:0'

    ckpt_path = '/home/manelguz/mcv-m6-2023-team3/tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

    # Configure the model for inference, starting with the default options
    import copy
    nn_opts = copy.deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
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

    pred_labels = np.array(nn.predict_from_img_pairs(img_pairs, batch_size=1)).squeeze()
    return pred_labels