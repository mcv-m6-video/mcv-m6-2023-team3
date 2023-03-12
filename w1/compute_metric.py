import random



def calculate_intersect_1d(gt_bb, predict_bb):
    min_side = gt_bb
    max_side = predict_bb
    if min_side[0] > max_side[0]:
        min_side, max_side = max_side, min_side
    return max(min_side[0]+min_side[1] - max_side[0],0)


def calculate_iou(gt_bb, predict_bb):
    intersect_width = calculate_intersect_1d([gt_bb[0],gt_bb[2]],[predict_bb[0],predict_bb[2]])
    intersect_height = calculate_intersect_1d([gt_bb[1],gt_bb[3]],[predict_bb[1],predict_bb[3]])
    intersect = intersect_width*intersect_height

    union = gt_bb[2]*gt_bb[3]+predict_bb[2]*predict_bb[3] - intersect
    return intersect/union


def calculate_noise_position(gt_bb_list, max_position):
    """ 
    Calculate AP and MIOU from the position distortion
    """
    total_iou = 0
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        noise_x = random.randint(0,max_position)
        noise_y = random.randint(0,max_position)
        predicted[0] += noise_x
        predicted[1] += noise_y

        total_iou += calculate_iou(gt_bb, predicted)
    
    miou = total_iou / len(gt_bb_list)

    return miou



def study_effect_noise(gt_bb_list, r_max_position = 50, r_max_size = 50, r_max_prob_generate = 0.5, r_max_prob_delete = 0.5):
    effect_noise = {}

    # change position
    effect_pos = []
    for max_position in range(0,r_max_position, 3):
        effect_pos.append([max_position, calculate_noise_position(gt_bb_list, max_position)])
    effect_noise['pos'] = effect_pos

    return effect_noise
