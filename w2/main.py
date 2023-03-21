# Import required packages
import os

from compute_metric import *
from models import GaussianModel, AdaptativeBackEstimator
from read_data import *

AICITY_DATA_PATH = '../AICity_data/train/S03/c010'

# Read gt file
ANOTATIONS_PATH = 'ai_challenge_s03_c010-full_annotation.xml'

task1_gifs = False

def task1():
    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussianModel = GaussianModel(
        path="../AICity_data/train/S03/c010/vdo.avi",
        colorSpace="gray")
    
    if task1_gifs:
        gaussianModel.get_video_rec_25()
        gaussianModel.get_video_rec_25_plot()


    # Load gt for(25-100)
    length = gaussianModel.find_length()
    gtInfo = parse_annotations(ANOTATIONS_PATH, isGT=True, startFrame=int(length * 0.25))

    # Model background
    gaussianModel.calculate_mean_std()

    # Separate foreground from background and calculate map
    alpha = 10
    print('Alpha:', alpha)
    predictionsInfo, num_bboxes = gaussianModel.model_foreground(alpha=alpha, gt=gtInfo)

    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)

def task2():
    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussianModel = GaussianModel(
        path="../AICity_data/train/S03/c010/vdo.avi",
        colorSpace="gray")

    # Load gt for(25-100)
    length = gaussianModel.find_length()
    gtInfo = parse_annotations(ANOTATIONS_PATH, isGT=True, startFrame=int(length * 0.25))

    # Model background
    gaussianModel.calculate_mean_std()

    # Separate foreground from background and calculate map
    alpha = 4
    print('Alpha:', alpha)
    predictionsInfo, num_bboxes = gaussianModel.model_foreground_Adaptive(alpha=alpha, rho=0.005)

    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)
"""
def task2():
    random_search = False
    # gt_bb = load_bb(os.path.join(AICITY_DATA_PATH, "gt/gt.txt"))
    print("Load video")
    # Load the video
    video_data = VideoData(
        video_path="../AICity_data/train/S03/c010/01_vdo.avi")

    # Load gt for(25-100)
    length = video_data.get_number_frames()
    video_data_train = video_data.conver_slice_to_grayscale(0, int(length * 0.25))
    # video_data_test = video_data.conver_slice_to_grayscale(int(length*0.25), length)

    print("pasrse Annotations")
    gtInfo = parse_annotations(path="ai_challenge_s03_c010-full_annotation.xml", isGT=True,
                               startFrame=int(length * 0.25))

    print("Estimation")
    roi = cv2.imread(os.path.join(AICITY_DATA_PATH, 'roi.jpg'), cv2.IMREAD_GRAYSCALE)
    bckg_estimator = AdaptativeBackEstimator(roi, (video_data.height, video_data.width, video_data.channels))

    print("Foreground")
    bckg_estimator.train(video_data_train)
    del video_data_train

    print("Evaluation")
    if random_search:
        rSearchIterations = 5

        params = {}
        params['alpha'] = np.arange(3, 10, 0.5)
        params['rho'] = np.arange(0.005, 0.5, 0.005)

        # Generation of parameter candidates
        # randomParameters = list(ParameterSampler(params, n_iter=rSearchIterations))
        bestIteration = 0
        bestScore = 0
        # Uncomment to run once with best parameters
        randomParameters = [{'alpha': 9.5, 'rho': 0.42}]

        for i, combination in enumerate(randomParameters):
            print("Trial " + str(i) + " out of " + str(len(randomParameters)))
            print("Testing the parameters:")
            print(combination)

            num_bboxes = 0
            predictionsInfo = []
            detections = []
            for idx in range(int(length * 0.25), length, 1):
                frame = video_data.convert_frame_by_idx(idx)
                detection, foreground_mask = bckg_estimator.evaluate(frame)
                detections.append(detections)
                predictionsInfo.append(({"frame": idx, "bbox": np.array(detection)}))
                num_bboxes = num_bboxes + len(detection)

            rec, prec, mAP, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
            print('mAP:', mAP)
            print('Mean IoU:', meanIoU)

            # now restore stdout function
            # sys.stdout = sys.__stdout__

            if mAP > best_map:
                best_map = mAP
                best_tieration = i

        print("The best mAP was: " + str(best_map))
        print("The best params are:" + randomParameters[best_tieration])
    else:
        num_bboxes = 0
        predictionsInfo = []
        print("no random")
        for idx in range(int(length * 0.25), length, 1):
            print("processing")
            frame = video_data.convert_frame_by_idx(idx)
            # Run with the best params found 
            detection, foreground_mask = bckg_estimator.evaluate(frame, rho=0.42, alpha=9.5)
            predictionsInfo.append(({"frame": idx, "bbox": np.array(detection)}))
            num_bboxes = num_bboxes + len(detection)

        rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
        print('mAP:', ap)
        print('Mean IoU:', meanIoU)
"""

def task4():
    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussianModel = GaussianModel(path="../AICity_data/train/S03/c010/vdo.avi", colorSpace="hs")

    # Load gt for(25-100)
    length = gaussianModel.find_length()
    gtInfo = parse_annotations(path="ai_challenge_s03_c010-full_annotation.xml", isGT=True, startFrame=int(length * 0.25))

    # Model background
    gaussianModel.calculate_mean_std()

    # Separate foreground from background and calculate map
    alpha = [8]
    print('Alpha:', alpha)
    predictionsInfo, num_bboxes = gaussianModel.model_foreground_Adaptive(alpha=alpha, rho=0.005)
    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)


task4()
