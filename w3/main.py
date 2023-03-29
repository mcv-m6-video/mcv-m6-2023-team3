# Import required packages
import os
import sys
from compute_metric import *
from kalman import task2_2
from read_data import *
import matplotlib.pyplot as plt
import imageio
from tqdm import trange
from sklearn.model_selection import ParameterSampler

AICITY_DATA_PATH = 'AICity_data/train/S03/c010'

# Read gt file
ANOTATIONS_PATH = 'ai_challenge_s03_c010-full_annotation.xml'


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 2.2

    if task == 1.1:
        task1_1()
    elif task == 1.3:
        task1_3()
    elif task == 1.4:
        task1_4()
    elif task == 2.1:
        task2_1()
    elif task == 2.2:
        task2_2()

if __name__ == "__main__":
    main(sys.argv)