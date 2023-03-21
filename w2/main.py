# Import required packages
import sys
import os


import cv2

from compute_metric import *
from models import GaussianModel, AdaptativeBackEstimator
from read_data import *


AICITY_DATA_PATH = '../AICity_data/train/S03/c010'

# Read gt file
ANOTATIONS_PATH = 'ai_challenge_s03_c010-full_annotation.xml'


def task1():

    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussianModel = GaussianModel(path="AICity_data/train/S03/c010/vdo.avi", colorSpace="gray")

    # Load gt for(25-100)
    length = gaussianModel.find_length()
    gtInfo = parse_annotations("../" + ANOTATIONS_PATH, isGT=True, startFrame=int(length*0.25))

    # Model background
    gaussianModel.calculate_mean_std()

    # Separate foreground from background and calculate map
    alpha = [6]
    print('Alpha:', alpha)
    predictionsInfo, num_bboxes = gaussianModel.model_foreground(alpha=alpha)
    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)

def task2():
    #gt_bb = load_bb(os.path.join(AICITY_DATA_PATH, "gt/gt.txt"))
    
    #Load the video
    video_data = VideoData((os.path.join(AICITY_DATA_PATH, "vdo.avi")))

    # Load gt for(25-100)
    length = video_data.get_number_frames()
    video_data_train = video_data.conver_slice_to_grayscale(0, int(length*0.25))
    video_data_test = video_data.conver_slice_to_grayscale(int(length*0.25), length)

    gtInfo = parse_annotations(os.path.join("..", ANOTATIONS_PATH), isGT=True, startFrame=int(length*0.25))

    roi = cv2.imread(os.path.join(AICITY_DATA_PATH,'roi.jpg'), cv2.IMREAD_GRAYSCALE)
    bckg_estimator = AdaptativeBackEstimator(roi,(video_data.height, video_data.width))

    bckg_estimator.train(video_data_train)
    predictionsInfo, num_bboxes = bckg_estimator.evaluate(video_data_test)
    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)

def task4():
    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussianModel = GaussianModel(path="AICity_data/train/S03/c010/vdo.avi", colorSpace="rgb")

    # Load gt for(25-100)
    length = gaussianModel.find_length()
    gtInfo = parse_annotations(ANOTATIONS_PATH, isGT=True, startFrame=int(length*0.25))

    # Model background
    gaussianModel.calculate_mean_std()

    # Separate foreground from background and calculate map
    alpha = [6]
    print('Alpha:', alpha)
    predictionsInfo, num_bboxes = gaussianModel.model_foreground(alpha=alpha)
    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)
task4()
