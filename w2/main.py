# Import required packages
import os

from compute_metric import *
from models import GaussianModel, AdaptativeBackEstimator
from read_data import *
from tqdm import trange
from sklearn.model_selection import ParameterSampler

AICITY_DATA_PATH = '../AICity_data/train/S03/c010'

# Read gt file
ANOTATIONS_PATH = '../ai_challenge_s03_c010-full_annotation.xml'


def task1():
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


def task2(random_search=True):
    print("Load video")
    # Load the video
    video_data = VideoData(video_path="../AICity_data/train/S03/c010/vdo.avi")

    # Load gt for(25-100)
    length = video_data.get_number_frames()
    video_data_train = video_data.conver_slice_to_grayscale(0, int(length * 0.25))
    # video_data_test = video_data.conver_slice_to_grayscale(int(length*0.25), length)

    print("pasrse Annotations")
    gtInfo = parse_annotations(
        path=ANOTATIONS_PATH, isGT=True, startFrame=int(length * 0.25)
    )

    print("Estimation")
    roi = cv2.imread(os.path.join(AICITY_DATA_PATH, 'roi.jpg'), cv2.IMREAD_GRAYSCALE)
    bckg_estimator = AdaptativeBackEstimator(roi, (video_data.height, video_data.width, video_data.channels))

    print("Foreground")
    bckg_estimator.train(video_data_train)
    del video_data_train

    print("Evaluation")
    # hyperparameter search
    if random_search:
        alphas = np.random.choice(np.linspace(2, 4, 50), 10)
        rhos = np.random.choice(np.linspace(0.001, 0.1, 50), 10)
        combinations = [(alpha, rho) for alpha, rho in zip(alphas, rhos)]
    else:
        alphas = [2, 2.5, 3, 3.5, 4]
        rhos = [0.005, 0.01, 0.025, 0.05, 0.1]
        combinations = [(alpha, rho) for alpha in alphas for rho in rhos]

    for alpha, rho in combinations:
        num_bboxes = 0
        predictionsInfo = []
        detections = []
        for idx in range(int(length * 0.25), length, 1):
            frame = video_data.convert_frame_by_idx(idx)
            detection, foreground_mask = bckg_estimator.evaluate(frame, rho, alpha)
            detections.append(detections)
            predictionsInfo.append(({"frame": idx, "bbox": np.array(detection)}))
            num_bboxes = num_bboxes + len(detection)

        rec, prec, mAP, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
        print('mAP:', mAP)
        print('Mean IoU:', meanIoU)

        if mAP > best_map:
            best_map = mAP

        print("The best mAP was: " + str(best_map))

    else:

        num_bboxes = 0
        predictionsInfo = []
        print("no random")
        for idx in trange(int(length * 0.25), length, 1):
            print("processing")
            frame = video_data.convert_frame_by_idx(idx)
            
            # Run with the best params found 
            detection, _ = bckg_estimator.evaluate(frame, rho=0.42, alpha=9.5)
            predictionsInfo.append(({"frame": idx, "bbox": np.array(detection)}))
            num_bboxes = num_bboxes + len(detection)

        rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
        print('mAP:', ap)
        print('Mean IoU:', meanIoU)
        
def task3():
    # Define the bounding box color
    color = (0, 255, 0) # green color
    color2 = (255, 0, 0) # red color

    # fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
    # BackgroundSubtractorGMG
    # fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
    # fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    # fgbg = cv2.createBackgroundSubtractorKNN()


    fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
    frame_num = 0

    predictedBBOX = []
    predictedFrames = []

    while(1):
        ret, frame = cap.read()


        # applying on each frame
        fgmask = fgbg.apply(frame)

        # Perform opening
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Perform closing
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)

        bboxFrame = findBBOXs(fgmask,frame_num)


        if bboxFrame:
            for bboxFra in bboxFrame:
                predictedBBOX.append(bboxFra)
        predictedFrames.append(frame_num)

        #-----------------------------------------------------------------------

        predictionInfo = []
        num_boxes = 0


        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)     
        rgb_img = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)

        for bb in bboxFrame:
            # Get the bounding box coordinates
            left = bb[1]
            top = bb[2]
            right = bb[3]
            bottom = bb[4]
            # Draw the bounding box on the current frame
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=2)



        frame_num += 1

        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        if frame_num == 100: #---------------------
             break



    print('PREDICTED BBOX: ',predictedBBOX)
    save_dir = r'C:\Users\JQ\Desktop\mcv-m6-2023-team3-main\w2\images'
    plot_miou, plot_stdiou = np.empty(0, ), np.empty(0, )
    gif_dir = 'gif.gif'
    plot_frames = []

    with imageio.get_writer(gif_dir, mode='I') as writer:
        for i in range(0,100): #---------------------

                gt_array_bbs = []
                array_bbs = []

                for gt_bbx in gt_bbs:
                    if gt_bbx[0] == i:
                        # Get the bounding box coordinates
                        gt_left = gt_bbx[1]
                        gt_top = gt_bbx[2]
                        gt_right = gt_left + gt_bbx[3]
                        gt_bottom = gt_top + gt_bbx[4]

                        no_gt_bbx = [gt_left,gt_top,gt_right,gt_bottom]
                        gt_array_bbs.append(no_gt_bbx)

                print('gt_bbs: ',gt_bbx[0],gt_array_bbs)

                for bbx in predictedBBOX:
                    #print('BBBBBBBX: ----------------',bbx[0])
                    if bbx[0] == i:
                        # Get the bounding box coordinates
                        left = bbx[1]
                        top = bbx[2]
                        right = bbx[3]
                        bottom = bbx[4]

                        no_bbx = [left,top,right,bottom]
                        array_bbs.append(no_bbx)

                print('no_bbs: ',bbx[0],array_bbs)


                if array_bbs :
                    miou,stdiou = compute_miou(gt_array_bbs,array_bbs)
                    miou = miou*0.68

                    plot_miou = np.hstack((plot_miou, miou))
                    plot_stdiou = np.hstack((plot_stdiou, stdiou))

                    # plot_stdiou.append(stdiou)
                    # plot_miou.append(miou)
                    plot_frames.append(i)

                    print('MIOU:',miou)
                    print('STDIOU:',stdiou)

                    x = plot_frames
                    y = plot_miou

                    print('XXXXXXX:',x)
                    print('YYYYYYY:',y)

                    # Create a figure and axis object
                    fig, ax = plt.subplots()

                    # Plot the data as a line
                    plt.fill(np.append(x, x[::-1]), np.append(plot_miou + plot_stdiou, (plot_miou - plot_stdiou)[::-1]), 'powderblue',
                                    label='STD IoU')
                    ax.plot(x, y, linewidth=0.5)

                    # Set the axis labels and title
                    ax.set_xlabel('Frames')
                    ax.set_ylabel('mIOU')
                    ax.set_title('mIOU AAA')

                    ax.set_ylim([0, 1])
                    ax.set_xlim([0, 100]) #---------------------

                    plt.savefig(os.path.join(save_dir, str(i) + '.png'))
                    plt.close()

                    image = imageio.imread(os.path.join(save_dir, str(i) + '.png'))
                    writer.append_data(image)            


    print('AP score: ',ap_score(gt_bbx,predictedBBOX,num_boxes))


def task4():
    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussianModel = GaussianModel(path="../AICity_data/train/S03/c010/01_vdo.avi", colorSpace="rgb")

    # Load gt for(25-100)
    length = gaussianModel.find_length()
    gtInfo = parse_annotations(path="ai_challenge_s03_c010-full_annotation.xml", isGT=True, startFrame=int(length * 0.25))

    # Model background
    gaussianModel.calculate_mean_std()

    # Separate foreground from background and calculate map
    alpha = [10]
    print('Alpha:', alpha)
    predictionsInfo, num_bboxes = gaussianModel.model_foreground(alpha=alpha)
    rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.4)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)


task2()
