import sys
import os
import cv2

import matplotlib.pyplot as plt
from OpticalFlowToolkit.lib import flowlib

from plot_data import *
from load_input import *
from compute_metric import *

path = './w1/AICity_data/train/S03/c010'

PATH_IMG_PRED = "./w1/data_w1/pred"
PATH_IMG_GT = "./w1/data_w1/flow_noc"
PATH_IMG = "./w1/data_w1/img"
IMG_LIST = ["000045_10.png", "000157_10.png"]


def main(argv, path):


    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 4

    if task == 1.1:
        frame = 420
        path_gt = os.path.join(path, "gt/gt.txt")
        img_frm = load_frames(os.path.join(path, "vdo.avi"), frame)
        gt_bb = load_bb(path_gt)
        
        frame_ini_pos = 0
        frame_last_pos = 0


        # Get the first and the last BB that has that frame
        while len(gt_bb) > frame_ini_pos and gt_bb[frame_ini_pos][0] != frame: frame_ini_pos += 1
        frame_last_pos = frame_ini_pos + 1
        while len(gt_bb) > frame_last_pos and gt_bb[frame_last_pos][0] == frame: frame_last_pos += 1
        gt_bb_frame = gt_bb[frame_ini_pos:frame_last_pos]
        predict = calculate_predict_generate(gt_bb_frame, 50)
        plot_bb(img_frm, gt_bb_frame,predict)

        

        study_noise = study_effect_noise(gt_bb)
        plot_ap_miou(study_noise['pos'], title="Displacement", xlabel="Max distance")
        plot_ap_miou(study_noise['size'], title="Resize", xlabel="Max distance")
        plot_ap_miou(study_noise['gen'], title="Generate", xlabel="Probability")
        plot_ap_miou(study_noise['del'], title="Delete", xlabel="Probability")

    if task == 4:

        img = cv2.imread(os.path.join(PATH_IMG,"000045_10.png"))
        flow_gt = flowlib.read_flow(os.path.join(PATH_IMG_GT,"000045_10.png"))
        flow_predict = flowlib.read_flow(os.path.join(PATH_IMG_PRED,"LKflow_000045_10.png"))
        plot_optical_flow(img, flow_gt)
        plot_optical_flow(img, flow_predict)
        
                








if __name__ == "__main__":
    main(sys.argv, path)