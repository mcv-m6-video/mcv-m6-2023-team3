# Import required packages
import os
import sys
from compute_metric import *
from kalman import task2_2
from overlap_tracking import task2_1
from read_data import *
import matplotlib.pyplot as plt
import imageio
from tqdm import trange
from sklearn.model_selection import ParameterSampler
from inference_video_yolov5 import yolo5_inf
from yolov8_train import task1_3
from kfold_cross_abc import strategy_a, strategy_b, strategy_c



AICITY_DATA_PATH = 'AICity_data/train/S03/c010'

# Read gt file
ANOTATIONS_PATH = 'ai_challenge_s03_c010-full_annotation.xml'


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 2.2

    if task == 1.1:
        yolo5_inf()
    elif task == 1.3:
        task1_3()
    elif task == 1.4:
        X = np.arange(255)

        strategy_a(X)
        strategy_b(X)
        strategy_c(X)
    elif task == 2.1:
        task2_1(path="", video_path="../AICity_data/train/S03/c010/01_vdo.avi")
    elif task == 2.2:
        task2_2()

if __name__ == "__main__":
    main(sys.argv)