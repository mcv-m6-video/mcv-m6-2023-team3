import os
import time
import sys

sys.path.append("../")

import cv2
import numpy as np
from opticalflow import compute_pyflow, compute_lukas_kanade, compute_epicflow
from OpticalFlowToolkit.lib import flowlib

from compute_metric import calculate_msen, calculate_pepn
from block_matching import compute_block_matching

ERROR_THRESH = 3


def task11(gridSearch=True, distance='cv2.TM_CCORR_NORMED'):
    # distance = 'cv2.TM_SQDIFF_NORMED'
    # distance = 'cv2.TM_CCORR_NORMED'

    # Read gt file
    im1 = cv2.imread('./data_w1/img/000045_10.png', 0)
    im2 = cv2.imread('./data_w1/img/000045_11.png', 0)
    flow_gt = flowlib.read_flow("./data_w1/flow_noc/000045_10.png")
    if gridSearch:
        # motion = ['forward','backward']# uncomment when not using the 3d plot
        motion = ['backward']
        blockSize = [4, 8, 16, 32, 64]
        searchAreas = np.array(blockSize) * 2 + np.array(blockSize)

    else:
        motion = ['backward']
        blockSize = [32]  # 32
        searchAreas = [96]  # 96

    all_msen = np.zeros((len(blockSize), len(searchAreas)))
    all_pepn = np.zeros((len(blockSize), len(searchAreas)))
    minerr = 10000
    start_time = []
    end_time = []
    pepns = []
    msens = []
    for m in motion:
        for i, bs in enumerate(blockSize):
            # quantStep = [int(bs/2),bs]
            quantStep = [int(bs / 2)]
            for q in quantStep:
                start_time.append(time.time())
                for j, sa in enumerate(searchAreas):
                    predicted_flow = compute_block_matching(im1, im2, m, sa, bs, distance, q)
                    # plotOf = PlotOF()
                    # utils.plot_module(predicted_flow)
                    # if m == 'forward':
                    #     plotOf.plotArrowsOP(predicted_flow, 10, im2)
                    # else:
                    #      plotOf.plotArrowsOP(predicted_flow, 10, im1)
                    msen = calculate_msen(predicted_flow, flow_gt, th=ERROR_THRESH)
                    pepn = calculate_pepn(predicted_flow, flow_gt, th=ERROR_THRESH)
                    pepns.append(pepn)
                    msens.append(msen)
                    all_msen[i, j] = msen
                    all_pepn[i, j] = pepn
                    print('Motion: ', m)
                    print('BS: ', bs)
                    print('SA: ', sa)
                    print('msen: ', msen)
                    print('pepn: ', pepn)
                    errsum = msen + pepn
                    if errsum < minerr:
                        bestQ = q
                        minerr = errsum
                        bestArea = sa
                        bestBlock = bs
                        bestMsen = msen
                        bestPepn = pepn
                end_time.append(time.time())
    if gridSearch:
        print('Best BS', bestBlock)
        print('Best AS', bestArea)
        print('Best Q', bestQ)
        print('Best pepn', bestPepn)
        print('Best msen', bestMsen)


    else:
        compTime = [end_time[t] - start_time[t] for t in range(len(start_time))]
        print(compTime)




def task1_2(method="pyflow"):

    img_prev = cv2.imread('../data_w1/img/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('../data_w1/img/000045_11.png', cv2.IMREAD_GRAYSCALE)
    gt_flow = flowlib.read_flow('../data_w1/flow_noc/000045_10.png')

    if method == 'pyflow':
        pred_flow = compute_pyflow(img, img_prev)

    elif method == 'kanade':
        pred_flow = compute_lukas_kanade(img, img_prev)
    elif method == 'epic':
        pred_flow = compute_epicflow(img, img_prev)


    msen = calculate_msen(gt_flow, pred_flow)
    print("The msen for the img: " + str(img) + " is: " + str(msen))
    pepn = calculate_pepn(gt_flow, pred_flow, ERROR_THRESH)
    print("The pepn for the img: " + str(img) + " is: " + str(pepn))


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.1

    if task == 1.1:
        task11()
    elif task == 1.2:
        task1_2()

if __name__ == "__main__":
    main(sys.argv)