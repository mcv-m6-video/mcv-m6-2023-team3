import os
import time
import sys

sys.path.append("../")

import cv2
import numpy as np
from epicflow import compute_epicflow
from opticalflow import compute_pyflow, compute_lukas_kanade
from OpticalFlowToolkit.lib import flowlib

from compute_metric import calculate_msen, calculate_pepn
from block_matching import compute_block_matching

ERROR_THRESH = 3

from plot_data import plot_3D, plot_optical_flow

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
        blockSize = [4]
        searchAreas = np.array(blockSize) * 2 + np.array(blockSize)

    else:
        motion = ['backward']
        blockSize = [32]  # 32
        searchAreas = [96]  # 96

    all_msen = np.zeros((len(blockSize), len(searchAreas)))
    all_pepn = np.zeros((len(blockSize), len(searchAreas)))
    all_time = np.zeros((len(blockSize), len(searchAreas)))
    minerr = 10000
    start_time = []
    end_time = []
    pepns = []
    msens = []
    plot_metrics = True
    best_predicted_flow = None
    for m in motion:
        for i, bs in enumerate(blockSize):
            # quantStep = [int(bs/2),bs]
            quantStep = [int(bs / 2)]
            for q in quantStep:
                start_time.append(time.time())
                for j, sa in enumerate(searchAreas):
                    start = time.time()
                    predicted_flow = compute_block_matching(im1, im2, m, sa, bs, distance, q)
                    # plotOf = PlotOF()
                    # utils.plot_module(predicted_flow)
                    # if m == 'forward':
                    #     plotOf.plotArrowsOP(predicted_flow, 10, im2)
                    # else:
                    #      plotOf.plotArrowsOP(predicted_flow, 10, im1)
                    msen = calculate_msen(flow_gt, predicted_flow, th=ERROR_THRESH)
                    pepn = calculate_pepn(flow_gt, predicted_flow, th=ERROR_THRESH)
                    pepns.append(pepn)
                    msens.append(msen)

                    all_msen[i, j] = msen
                    all_pepn[i, j] = pepn
                    print('Motion: ', m)
                    print('BS: ', bs)
                    print('SA: ', sa)
                    print('Q: ', q)
                    print('msen: ', msen)
                    print('pepn: ', pepn)
                    errsum = msen + pepn
                    if errsum < minerr:
                        best_predicted_flow = predicted_flow
                        bestQ = q
                        minerr = errsum
                        bestMotion = m
                        bestArea = sa
                        bestBlock = bs
                        bestMsen = msen
                        bestPepn = pepn
                    end = time.time()
                    all_time[i,j] = end - start
                end_time.append(time.time())
    if plot_metrics:
        plot_optical_flow(im2, best_predicted_flow)
        plot_3D(blockSize, searchAreas, all_msen, 'Block Size', 'Search Areas', 'MSEN')
        plot_3D(blockSize, searchAreas, all_pepn, 'Block Size', 'Search Areas', 'PEPN')
        plot_3D(blockSize, searchAreas, all_time, 'Block Size', 'Search Areas', 'Execution time')
    if gridSearch:
        print('Best M', bestMotion)
        print('Best BS', bestBlock)
        print('Best AS', bestArea)
        print('Best Q', bestQ)
        print('Best pepn', bestPepn)
        print('Best msen', bestMsen)


    else:
        compTime = [end_time[t] - start_time[t] for t in range(len(start_time))]
        print(compTime)




def task1_2(method="pyflow"):

    def read_img(color="gray"):
        if color == "gray":
            img_prev = cv2.imread('../data_w1/img/000045_10.png', cv2.IMREAD_GRAYSCALE)
            img = cv2.imread('../data_w1/img/000045_11.png', cv2.IMREAD_GRAYSCALE)
        elif color == "rgb":
            img_prev = cv2.imread('../data_w1/img_color/000045_10.png')
            img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB)
            img = cv2.imread('../data_w1/img_color/000045_11.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError
        
        return img, img_prev


    gt_flow = flowlib.read_flow('../data_w1/flow_noc/000045_10.png')

    if method == 'pyflow':
        img, img_prev = read_img(color="gray")
        pred_flow = compute_pyflow(img, img_prev)

    elif method == 'kanade':
        img, img_prev = read_img(color="gray")
        pred_flow = compute_lukas_kanade(img, img_prev)
    elif method == 'epic':
        img, img_prev = read_img(color="rgb")
        pred_flow = compute_epicflow(img, img_prev)


    msen = calculate_msen(gt_flow, pred_flow)
    print("The msen for the img: " + str(img) + " is: " + str(msen))
    pepn = calculate_pepn(gt_flow, pred_flow, ERROR_THRESH)
    print("The pepn for the img: " + str(img) + " is: " + str(pepn))


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.2

    if task == 1.1:
        task11()
    elif task == 1.2:
        task1_2()

if __name__ == "__main__":
    main(sys.argv)