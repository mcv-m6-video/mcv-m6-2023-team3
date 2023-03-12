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
 
        path = os.path.join(path, "gt/gt.txt")
        gt_bb = load_bb(path)
        study_noise = study_effect_noise(gt_bb)
        study_noise_pos = study_noise['pos']
        plot_ap_miou(study_noise_pos)








if __name__ == "__main__":
    main(sys.argv, path)