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
        plot_ap_miou(study_noise['pos'], title="Displacement", xlabel="Max distance")
        plot_ap_miou(study_noise['size'], title="Resize", xlabel="Max distance")
        plot_ap_miou(study_noise['gen'], title="Generate", xlabel="Probability")
        plot_ap_miou(study_noise['del'], title="Delete", xlabel="Probability")









if __name__ == "__main__":
    main(sys.argv, path)