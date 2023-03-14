import sys
import os
sys.path.append("../")


#https://github.com/liruoteng/OpticalFlowToolkit.git
from OpticalFlowToolkit.lib import flowlib


from plot_data import *
from load_input import *
from compute_metric import *

path = './w1/AICity_data/train/S03/c010'
PATH_IMG_PRED = "../data_w1/pred/"
PATH_IMG_GT = "../data_w1/flow_noc"
PATH_IMG = "../data_w1/img"
IMG_LIST = ["000045_10.png", "000157_10.png"]



def main(argv, path):
    print(path)

    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.1

    if task == 1.1:
 
        path = os.path.join(path, "gt/gt.txt")
        gt_bb = load_bb(path)
        study_noise = study_effect_noise(gt_bb)
        plot_ap_miou(study_noise['pos'], title="Displacement", xlabel="Max distance")
        plot_ap_miou(study_noise['size'], title="Resize", xlabel="Max distance")
        plot_ap_miou(study_noise['gen'], title="Generate", xlabel="Probability")
        plot_ap_miou(study_noise['del'], title="Delete", xlabel="Probability")

    elif task == 3.1:
        for img in IMG_LIST:
            img_pred_path = os.path.join(PATH_IMG_PRED, "LKflow_" + img)
            img_gt_path = os.path.join(PATH_IMG_GT, img)
            # Function from ://github.com/liruoteng/OpticalFlowToolkit.git
            pred_flow = flowlib.read_flow(img_pred_path)
            gt_flow = flowlib.read_flow(img_gt_path)
            msen = calculate_msen(gt_flow, pred_flow)
            print("The msen for the img: " + str(img) + " is: " + str(msen))










if __name__ == "__main__":
    main(sys.argv, path)