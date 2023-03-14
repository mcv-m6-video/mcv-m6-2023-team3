import sys
import os


from plot_data import *
from load_input import *
from compute_metric import *

path = './w1/AICity_data/train/S03/c010'



def main(argv, path):
    print(path)

    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.1

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











if __name__ == "__main__":
    main(sys.argv, path)