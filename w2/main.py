# Import required packages
import sys
import os
import pickle


import cv2
from sklearn.model_selection import ParameterSampler

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
    random_search = False
    #gt_bb = load_bb(os.path.join(AICITY_DATA_PATH, "gt/gt.txt"))
    
    #Load the video
    video_data = VideoData((os.path.join(AICITY_DATA_PATH, "vdo.avi")))

    # Load gt for(25-100)
    length = video_data.get_number_frames()
    video_data_train = video_data.conver_slice_to_grayscale(0, int(length*0.25))
    #video_data_test = video_data.conver_slice_to_grayscale(int(length*0.25), length)

    gtInfo = parse_annotations(os.path.join("..", ANOTATIONS_PATH), isGT=True, startFrame=int(length*0.25))

    roi = cv2.imread(os.path.join(AICITY_DATA_PATH,'roi.jpg'), cv2.IMREAD_GRAYSCALE)
    bckg_estimator = AdaptativeBackEstimator(roi,(video_data.height, video_data.width, video_data.channels))

    bckg_estimator.train(video_data_train)
    del video_data_train

    if random_search:
        rSearchIterations = 5

        params = {}
        params['alpha'] = np.arange(3, 10, 0.5)
        params['rho'] = np.arange(0.005, 0.5, 0.005)

        # Generation of parameter candidates
        randomParameters = list(ParameterSampler(params, n_iter=rSearchIterations))
        bestIteration = 0
        bestScore = 0
        #Uncomment to run once with best parameters
        #randomParameters = [{'alpha': 9.5, 'rho': 0.42}]

        for i, combination in enumerate(randomParameters):
            print("Trial " + str(i) + " out of " + str(len(randomParameters)))
            print("Testing the parameters:")
            print(combination)

            num_bboxes = 0
            predictionsInfo = []
            detections = []
            for idx in range(int(length*0.25), length, 1):
                frame = video_data.convert_frame_by_idx(idx) 
                detection , foreground_mask= bckg_estimator.evaluate(frame)
                detections.append(detections)
                predictionsInfo.append(({"frame": idx, "bbox": np.array(detection)}))
                num_bboxes = num_bboxes + len(detection)
            
            rec, prec, mAP, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
            print('mAP:', mAP)
            print('Mean IoU:', meanIoU)

            # now restore stdout function
            #sys.stdout = sys.__stdout__

            if mAP > best_map:
                best_map = mAP
                best_tieration = i

        print("The best mAP was: " + str(best_map))
        print("The best params are: "randomParameters[best_tieration])
    else:
        num_bboxes = 0
        predictionsInfo = []
        for idx in range(int(length*0.25), length, 1):
            frame = video_data.convert_frame_by_idx(idx) 
            # Run with the best params found 
            detection , foreground_mask= bckg_estimator.evaluate(frame, rho=0.42, alpha=9.5)
            predictionsInfo.append(({"frame": idx, "bbox": np.array(detection)}))
            num_bboxes = num_bboxes + len(detection)
        
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
task2()
