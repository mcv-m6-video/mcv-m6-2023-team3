# Import required packages
from compute_metric import *
from models import GaussianModel
from read_data import *


def task1():
    # Read gt file
    path = 'ai_challenge_s03_c010-full_annotation.xml'

    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussianModel = GaussianModel(path="../AICity_data/train/S03/c010/01_vdo.avi", colorSpace="gray")

    # Load gt for(25-100)
    length = gaussianModel.find_length()
    gtInfo = parse_annotations(path, isGT=True, startFrame=int(length*0.25))

    # Model background
    gaussianModel.calculate_mean_std()

    # Separate foreground from background and calculate map
    alpha = [6]
    print('Alpha:', alpha)
    predictionsInfo, num_bboxes = gaussianModel.model_foreground(alpha=alpha)
    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)


task1()
