import numpy as np
import cv2
from tqdm import trange


def extractRotation_translation(prev_gray, vCapture):
    prev_goodPts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    # Read current frame
    available, current = vCapture.read()
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    current_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_goodPts, None)
    assert prev_goodPts.shape == current_pts.shape

    # remove invalid points
    idx = np.where(status == 1)[0]
    prev_goodPts = prev_goodPts[idx]
    current_pts = current_pts[idx]

    M, _ = cv2.estimateAffinePartial2D(prev_goodPts, current_pts)
    # Translation:
    dx = M[0, 2]
    dy = M[1, 2]

    # Rotation:
    a = np.arctan2(M[1, 0], M[0, 0])

    return dx, dy, a, current_gray


def stabilise_frame(transformations_smooth, i, f, w, h):
    dx = transformations_smooth[i, 0]
    dy = transformations_smooth[i, 1]
    a = transformations_smooth[i, 2]

    M = np.zeros((2, 3))
    M[0, 0] = np.cos(a)
    M[0, 1] = -np.sin(a)
    M[1, 0] = np.sin(a)
    M[1, 1] = np.cos(a)
    M[0, 2] = dx
    M[1, 2] = dy

    stabilised_f = cv2.warpAffine(f, M, (w, h))
    frame_out = cv2.hconcat([f, stabilised_f])

    return stabilised_f, frame_out


def calculate_adaptiveRegion_(centerBlockj, w, searchArea, boolAdapt):
    if boolAdapt:
        if centerBlockj < w / 2:
            init_j = max(centerBlockj - searchArea / 2, 0)
            offset = min(centerBlockj - searchArea / 2, 0)
            end_j = centerBlockj + searchArea / 2 - offset
        elif centerBlockj == w / 2:
            init_j = centerBlockj - searchArea / 2
            end_j = centerBlockj + searchArea / 2
        elif centerBlockj > w / 2:
            end_j = min(centerBlockj + searchArea / 2, w)
            offset = max(centerBlockj + searchArea / 2, w) - w
            init_j = centerBlockj - searchArea / 2 - offset
    else:
        init_j = max(centerBlockj - searchArea / 2, 0)
        end_j = min(centerBlockj + searchArea / 2, w)

    return init_j, end_j


def evaluation_metrics_(method, target, ref):
    if method != 'euclidean':
        method = eval(method)
        res = cv2.matchTemplate(target, ref, method)

    return res


def compute_block_matching(im1, im2, motion, searchArea, blockSize, method, q):
    if motion == 'forward':
        refIm = im1
        targetIm = im2
    elif motion == 'backward':
        refIm = im2
        targetIm = im1

    h, w = np.shape(refIm)
    predicted_flow = np.zeros((h, w, 3))

    for i in trange(0, h - blockSize, q):
        centerBlocki = int(i + blockSize / 2)
        init_i, end_i = calculate_adaptiveRegion_(centerBlocki, h, searchArea, False)

        for j in range(0, w - blockSize, q):
            centerBlockj = int(j + blockSize / 2)
            init_j, end_j = calculate_adaptiveRegion_(centerBlockj, w, searchArea, False)

            r = refIm[i:i + blockSize, j:j + blockSize]
            # t = targetIm[int(max(centerBlocki-searchArea/2,0)):int(min(centerBlocki+searchArea/2,h)),
            # int(max(centerBlockj-searchArea/2,0)):int(min(centerBlockj+searchArea/2,w))]
            t = targetIm[int(init_i):int(end_i), int(init_j):int(end_j)]
            result = evaluation_metrics_(method, t, r)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val < 1.0:
                ci = cj = int(searchArea / 2 - (blockSize / 2))
                if centerBlocki - searchArea / 2 < 0:
                    ci = ci + (centerBlocki - searchArea / 2)
                if centerBlockj - searchArea / 2 < 0:
                    cj = cj + (centerBlockj - searchArea / 2)

                if motion == 'forward':  # distance from the highest response to the center of the search space
                    flowVect = np.array(np.array(max_loc) - [cj, ci])
                else:
                    flowVect = np.array([cj, ci]) - np.array(max_loc)
            else:
                flowVect = [0, 0]

            predicted_flow[i:i + blockSize, j:j + blockSize] = np.array([flowVect[0], flowVect[1], 1])

    return predicted_flow
