import random
from utils import *
from sklearn.metrics import average_precision_score
import numpy as np


def calculate_intersect_1d(gt_bb, predict_bb):
    min_side = gt_bb
    max_side = predict_bb
    if min_side[0] > max_side[0]:
        min_side, max_side = max_side, min_side
    return min(max(min_side[0] + min_side[1] - max_side[0] + 1, 0), max_side[1])


def calculate_iou(gt_bb, predict_bb):
    intersect_width = calculate_intersect_1d([gt_bb[0], gt_bb[2]], [predict_bb[0], predict_bb[2]])
    intersect_height = calculate_intersect_1d([gt_bb[1], gt_bb[3]], [predict_bb[1], predict_bb[3]])
    intersect = intersect_width * intersect_height

    union = gt_bb[2] * gt_bb[3] + predict_bb[2] * predict_bb[3] - intersect
    return intersect / union


def frameIOU(boxA, boxB):
    # For each prediction, compute its iou over all the boxes in that frame
    xleft1, yleft1, xright1, yright1 = np.split(boxA, 4, axis=1)
    xleft2, yleft2, xright2, yright2 = np.split(boxB, 4, axis=1)

    # Calculate the intersection in the bboxes
    xmin = np.maximum(xleft1, np.transpose(xleft2))
    ymin = np.maximum(yleft1, np.transpose(yleft2))
    xmax = np.minimum(xright1, np.transpose(xright2))
    ymax = np.minimum(yright1, np.transpose(yright2))
    w = np.maximum(xmax - xmin + 1.0, 0.0)
    h = np.maximum(ymax - ymin + 1.0, 0.0)
    intersection = w * h

    # Calculate the Union in the bboxes
    areaboxA = (xright1 - xleft1 + 1.0) * (yright1 - yleft1 + 1.0)
    areaboxB = (xright2 - xleft2 + 1.0) * (yright2 - yleft2 + 1.0)
    union = areaboxA + np.transpose(areaboxB) - intersection

    # Calculate IOU
    iou = intersection / union
    maxScore = max(iou)
    index = np.argmax(iou)

    # Return the IOU, maxscore, index
    return iou, maxScore, index


def calculate_ap_prec_rec_confi(groundTruth, prediction, num_bboxes, overlapThreshold=0.5):
    # Sort by confidence
    pred_BB = []
    for i in range(len(prediction)):
        for i_bb in range(len(prediction[i]['bbox'])):
            pred_BB.append([i, prediction[i]['bbox'][i_bb], prediction[i]['score'][i_bb]])
    pred_bb_sorted = sorted(pred_BB, reverse=True, key=lambda elem: elem[2])

    # go down dets and mark TPs and FPs
    nd = num_bboxes
    true_positives = np.zeros(nd)
    false_positives = np.zeros(nd)

    for pred_bb in range(len(pred_bb_sorted)):
        # for box in bboxes_pred:
        frame_id = pred_bb_sorted[pred_bb][0]
        bbpred = np.array([pred_bb_sorted[pred_bb][1]]).astype(float)
        bbgt = groundTruth[frame_id]['bbox'].astype(float)

        iouScore, maxScore, index = frameIOU(bbgt, bbpred)

        if maxScore > overlapThreshold:
            if not groundTruth[frame_id]['already_detected'][index]:
                # We have detected an existing bbox in the gt
                groundTruth[frame_id]['already_detected'][index] = True
                true_positives[pred_bb] = 1.0
            else:
                false_positives[pred_bb] = 1.0
        else:
            false_positives[pred_bb] = 1.0

    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # Compute precision recall
    rec = true_positives / float(num_bboxes)

    # Avoid divide by zero in case the first detection matches a difficult
    prec = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute AP with the 11 point metric
    ap = calculate_ap(prec, rec)

    return rec, prec, ap


def calculate_ap(prec, rec):
    """
    Compute AP given precision and recall from pascal VOC approach.
    """

    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap


def calculate_ap_prec_rec_no_confi(gt_bb, predicted_bb, iou_threshold=0.5, iterations=10):
    """
    Computes AP from ground truth bounding boxes and predicted bounding boxes.
    """

    ap = 0.

    for _ in range(iterations):
        frame_ini_pos_gt = 0
        frame_last_pos_gt = 0
        frame_ini_pos_pr = 0
        frame_last_pos_pr = 0
        true_positives = np.zeros(len(predicted_bb))
        false_positives = np.zeros(len(predicted_bb))
        frame = 0
        while frame < gt_bb[-1][0]:
            frame_ini_pos_gt = frame_last_pos_gt
            frame_ini_pos_pr = frame_last_pos_pr

            frame = gt_bb[frame_ini_pos_gt][0]
            while len(gt_bb) > frame_last_pos_gt and gt_bb[frame_last_pos_gt][0] == frame: frame_last_pos_gt += 1
            while len(predicted_bb) > frame_last_pos_pr and predicted_bb[frame_last_pos_pr][
                0] == frame: frame_last_pos_pr += 1
            gt_frame = gt_bb[frame_ini_pos_gt:frame_last_pos_gt]
            predict_frame = predicted_bb[frame_ini_pos_pr:frame_last_pos_pr]

            ranked_predict = random_rank(predict_frame)

            gt_bb_copy = list(gt_frame)
            for i, box in enumerate(ranked_predict):
                iou_scores = [(pos, calculate_iou(box[1:], gt_box[1:])) for pos, gt_box in enumerate(gt_bb_copy)]
                val = max(iou_scores, key=lambda x: x[1])
                if val[1] >= iou_threshold:
                    gt_bb_copy[val[0]] = [-1, -1, -1, -1, -1]
                    true_positives[frame_ini_pos_pr + i] = 1
                else:
                    false_positives[frame_ini_pos_pr + i] = 1

        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        recall = true_positives / len(gt_bb)
        ap += (calculate_ap(precision, recall) / iterations)
    return ap, precision, recall


def calculate_noise_position(gt_bb_list, max_position):
    """ 
    Calculate AP and MIOU from the position distortion
    """
    total_iou = 0
    predict_list = []
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        noise_x = random.randint(0, max_position)
        noise_y = random.randint(0, max_position)
        predicted[1] += noise_x
        predicted[2] += noise_y

        predict_list.append(predicted)

        total_iou += calculate_iou(gt_bb[1:], predicted[1:])

    miou = total_iou / len(gt_bb_list)
    ap, _, _ = calculate_ap_prec_rec_no_confi(gt_bb_list, predict_list)
    return miou, ap


def calculate_noise_size(gt_bb_list, max_size):
    """ 
    Calculate AP and MIOU from the size distortion
    """
    total_iou = 0
    predict_list = []
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        noise_x = random.randint(0, max_size)
        noise_y = random.randint(0, max_size)
        predicted[3] += noise_x
        predicted[4] += noise_y

        predict_list.append(predicted)

        total_iou += calculate_iou(gt_bb, predicted)

    miou = total_iou / len(gt_bb_list)
    ap, _, _ = calculate_ap_prec_rec_no_confi(gt_bb_list, predict_list)
    return miou, ap


def calculate_predict_generate(gt_bb_list, prob_create, x_size=1920, y_size=1080, max_gen=10):
    """ 
    Calculate predictions from the generation distortion
      ----> x
      |
      |
    y V
    """
    predict_list = []
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        predict_list.append(predicted)

    for gt_bb in gt_bb_list:
        for _ in range(max_gen):
            if random.randint(0, 100) < prob_create:
                left = random.randint(0, x_size)
                top = random.randint(0, y_size)
                width = random.randint(left, x_size) - left
                height = random.randint(top, y_size) - top

                predict_list.append([gt_bb[0], left, top, width, height])

    return predict_list


def calculate_noise_generate(gt_bb_list, prob_create, x_size=1920, y_size=1080, max_gen=5):
    """ 
    Calculate AP and MIOU from the generation distortion
      ----> x
      |
      |
    y V
    """
    total_iou = 0
    predict_list = []
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        total_iou += calculate_iou(gt_bb[1:], predicted[1:])
        predict_list.append(predicted)

        for _ in range(max_gen):
            if random.randint(0, 100) < prob_create:
                left = random.randint(0, x_size)
                top = random.randint(0, y_size)
                width = random.randint(left, x_size) - left
                height = random.randint(top, y_size) - top

                total_iou += calculate_iou(gt_bb[1:], [left, top, width, height])

                predict_list.append([gt_bb[0], left, top, width, height])

    miou = total_iou / len(predict_list)
    ap, _, _ = calculate_ap_prec_rec_no_confi(gt_bb_list, predict_list)
    return miou, ap


def calculate_noise_delete(gt_bb_list, prob_delete):
    """ 
    Calculate AP and MIOU from the delete distortion
    """
    total_iou = 0
    predict_list = []
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        if random.randint(0, 100) >= prob_delete:
            total_iou += calculate_iou(gt_bb[1:], predicted[1:])
            predict_list.append(predicted)
    if len(predict_list) == 0:
        miou = 1.0
    else:
        miou = total_iou / len(predict_list)
    ap, _, _ = calculate_ap_prec_rec_no_confi(gt_bb_list, predict_list)
    return miou, ap


def study_effect_noise(gt_bb_list, r_max_position=100, r_max_size=100, r_max_prob_generate=100, r_max_prob_delete=100):
    effect_noise = {}

    # change position
    effect_pos = []
    for max_position in range(0, r_max_position, 3):
        effect_pos.append([max_position, calculate_noise_position(gt_bb_list, max_position)])
    effect_noise['pos'] = effect_pos

    effect_size = []
    for max_size in range(0, r_max_size, 3):
        effect_size.append([max_size, calculate_noise_size(gt_bb_list, max_size)])
    effect_noise['size'] = effect_size

    effect_generate = []
    for max_prob_generate in range(0, r_max_prob_generate, 3):
        effect_generate.append([max_prob_generate, calculate_noise_generate(gt_bb_list, max_prob_generate)])
    effect_noise['gen'] = effect_generate

    effect_delete = []
    for max_prob_delete in range(0, r_max_prob_delete, 3):
        effect_delete.append([max_prob_delete, calculate_noise_delete(gt_bb_list, max_prob_delete)])
    effect_noise['del'] = effect_delete

    return effect_noise


def calculate_msen(gt_flow, pred_flow):
    """
    Function to compute  the Mean Square Error in Non-occluded areas
    gt_flow: the ground thruth optical flow
    pred_flow: the predicted optical flow
    """
    # Get the mask of the valid points
    mask = gt_flow[:, :, 2] == 1

    # compute the error in du and dv
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    msen = np.mean(sqrt_error_masked)

    return msen


def calculate_pepn(gt_flow, pred_flow, th):
    """
    Function to compute the percentage of Erroneous Pixels in Non-occluded areas
    gt_flow: the ground thruth optical flow
    pred_flow: the predicted optical flow
    """

    # Get the mask of the valid points
    mask = gt_flow[:, :, 2] == 1

    # compute the error in du and dv
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]

    return np.sum(sqrt_error_masked > th) / len(sqrt_error_masked)
