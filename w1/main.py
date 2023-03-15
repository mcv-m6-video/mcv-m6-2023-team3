import sys
import os
import imageio
from read_data import parse_annotations, getPredictions
from compute_metric import calculate_ap_prec_rec_confi

sys.path.append("../")

# https://github.com/liruoteng/OpticalFlowToolkit.git
from OpticalFlowToolkit.lib import flowlib
import cv2
import matplotlib.pyplot as plt

from plot_data import *
from load_input import *
from compute_metric import *

path = './AICity_data/train/S03/c010'
PATH_IMG_PRED = "../data_w1/pred/"
PATH_IMG_GT = "../data_w1/flow_noc"
PATH_IMG = "../data_w1/img"
IMG_LIST = ["000045_10.png", "000157_10.png"]
ERROR_THRESH = 3

path_annotation = 'ai_challenge_s03_c010-full_annotation.xml'
detection_path = ['../AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
                  '../AICity_data/train/S03/c010/det/det_ssd512.txt',
                  '../AICity_data/train/S03/c010/det/det_yolo3.txt']

models = ['mask_rcnn', 'ssd512', 'yolo3']

PATH_IMG_PRED = "./w1/data_w1/pred"
PATH_IMG_GT = "./w1/data_w1/flow_noc"
PATH_IMG = "./w1/data_w1/img"
IMG_LIST = ["000045_10.png", "000157_10.png"]


def main(argv, path):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 3

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
        plot_bb(img_frm, gt_bb_frame, predict)
        predict = calculate_predict_position(gt_bb_frame, 50)
        plot_bb(img_frm, gt_bb_frame,predict)
        study_noise = study_effect_noise(gt_bb)
        plot_ap_miou(study_noise['pos'], title="Displacement", xlabel="Max distance")
        plot_ap_miou(study_noise['size'], title="Resize", xlabel="Max distance")
        plot_ap_miou(study_noise['gen'], title="Generate", xlabel="Probability")
        plot_ap_miou(study_noise['del'], title="Delete", xlabel="Probability")

    elif task == 1.2:
        for model in range(len(models)):
            groundTruthInfo, sortedFrames_gt = parse_annotations(path_annotation, isGT=True)
            predictionsInfo = getPredictions(detection_path[model], isGT=False)

            rec, prec, ap = calculate_ap_prec_rec_confi(groundTruthInfo, predictionsInfo,
                                                        num_bboxes=len(sortedFrames_gt),
                                                        overlapThreshold=0.5)
            print('Model:{} mAP:{}'.format(models[model], np.mean(ap)))
    elif task == 2:
        plot_miou = []
        plot_stdiou = []
        plot_frames = []
        
        plot_miou, plot_stdiou = np.empty(0, ), np.empty(0, )

        gif_dir = 'gif.gif'
        save_dir = './frame'
        gt_bbs,_,_ = parse_annotations(detection_path[0])
        bbs = load_bb(os.path.join(path, "gt/gt.txt"))
        with imageio.get_writer(gif_dir, mode='I') as writer:
            for i in range(1000,1800):
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

                print(gt_array_bbs)

                for bbx in bbs:
                    if bbx[0] == i:
                        # Get the bounding box coordinates
                        left = bbx[1]
                        top = bbx[2]
                        right = left + bbx[3]
                        bottom = top + bbx[4]

                        no_bbx = [left,top,right,bottom]
                        array_bbs.append(no_bbx)

                print(array_bbs)
                if array_bbs :
                    miou,stdiou = compute_miou(gt_array_bbs,array_bbs)

                    plot_miou = np.hstack((plot_miou, miou))
                    plot_stdiou = np.hstack((plot_stdiou, stdiou))
                    
                    # plot_stdiou.append(stdiou)
                    # plot_miou.append(miou)
                    plot_frames.append(i)

                    print('MIOU:',miou)
                    print('STDIOU:',stdiou)

                    x = plot_frames
                    y = plot_miou

                    # Create a figure and axis object
                    fig, ax = plt.subplots()

                    # Plot the data as a line
                    plt.fill(np.append(x, x[::-1]), np.append(plot_miou + plot_stdiou, (plot_miou - plot_stdiou)[::-1]), 'powderblue',
                                    label='STD IoU')
                    ax.plot(x, y, linewidth=0.5)

                    # Set the axis labels and title
                    ax.set_xlabel('Frames')
                    ax.set_ylabel('mIOU')
                    ax.set_title('mIOU mask RCNN')

                    ax.set_ylim([0, 1])
                    ax.set_xlim([1000, 1800])

                    plt.savefig(os.path.join(save_dir, str(i) + '.png'))
                    plt.close()

                    image = imageio.imread(os.path.join(save_dir, str(i) + '.png'))
                    writer.append_data(image)            

    elif task == 3:
        for img in IMG_LIST:
            img_pred_path = os.path.join(PATH_IMG_PRED, "LKflow_" + img)
            img_gt_path = os.path.join(PATH_IMG_GT, img)
            # Function from ://github.com/liruoteng/OpticalFlowToolkit.git
            pred_flow = flowlib.read_flow(img_pred_path)
            gt_flow = flowlib.read_flow(img_gt_path)
            msen = calculate_msen(gt_flow, pred_flow)
            print("The msen for the img: " + str(img) + " is: " + str(msen))
            pepn = calculate_pepn(gt_flow, pred_flow, ERROR_THRESH)
            print("The pepn for the img: " + str(img) + " is: " + str(pepn))
    elif task == 4:

        img = cv2.imread(os.path.join(PATH_IMG, "000045_10.png"))
        flow_gt = flowlib.read_flow(os.path.join(PATH_IMG_GT, "000045_10.png"))
        flow_predict = flowlib.read_flow(os.path.join(PATH_IMG_PRED, "LKflow_000045_10.png"))
        plot_optical_flow(img, flow_gt)
        plot_optical_flow(img, flow_predict)
        magnitudeOP(img, flow_gt)
        magnitudeOP(img, flow_predict)


if __name__ == "__main__":
    main(sys.argv, path)
