import random



def calculate_intersect_1d(gt_bb, predict_bb):
    min_side = gt_bb
    max_side = predict_bb
    if min_side[0] > max_side[0]:
        min_side, max_side = max_side, min_side
    return min(max(min_side[0]+min_side[1] - max_side[0]+1,0),max_side[1])


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
        noise_x = max_position #random.randint(0,max_position)
        noise_y = max_position #random.randint(0,max_position)
        predicted[0] += noise_x
        predicted[1] += noise_y

        total_iou += calculate_iou(gt_bb, predicted)
    
    miou = total_iou / len(gt_bb_list)

    return miou

def calculate_noise_size(gt_bb_list, max_size):
    """ 
    Calculate AP and MIOU from the size distortion
    """
    total_iou = 0
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        noise_x = random.randint(0,max_size)
        noise_y = random.randint(0,max_size)
        predicted[2] += noise_x
        predicted[3] += noise_y

        total_iou += calculate_iou(gt_bb, predicted)
    
    miou = total_iou / len(gt_bb_list)

    return miou


def calculate_noise_generate(gt_bb_list, prob_create, x_size=1920, y_size=1080):
    """ 
    Calculate AP and MIOU from the generation distortion
      ----> x
      |
      |
    y V
    """
    total_iou = 0
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        total_iou += calculate_iou(gt_bb, predicted)
    
    total_length = len(gt_bb_list)

    for gt_bb in gt_bb_list:
        if random.randint(0,100) < prob_create:
            left = random.randint(0,x_size)
            top = random.randint(0,y_size)
            width = random.randint(left,x_size) - left
            height = random.randint(top,y_size) - top
            total_iou += calculate_iou(gt_bb, [left,top,width,height])
            total_length += 1

    miou = total_iou / total_length

    return miou


def calculate_noise_delete(gt_bb_list, prob_delete):
    """ 
    Calculate AP and MIOU from the delete distortion
    """
    total_iou = 0
    for gt_bb in gt_bb_list:
        predicted = list(gt_bb)
        if random.randint(0,100) < prob_delete:
            total_iou += 0
        else:
            total_iou += calculate_iou(gt_bb, predicted)
    
    miou = total_iou / len(gt_bb_list)

    return miou


def study_effect_noise(gt_bb_list, r_max_position = 50, r_max_size = 50, r_max_prob_generate = 100, r_max_prob_delete = 100):
    effect_noise = {}

    # change position
    effect_pos = []
    for max_position in range(0,r_max_position, 3):
        effect_pos.append([max_position, calculate_noise_position(gt_bb_list, max_position)])
    effect_noise['pos'] = effect_pos

    effect_size = []
    for max_size in range(0,r_max_size, 3):
        effect_size.append([max_size, calculate_noise_size(gt_bb_list, max_size)])
    effect_noise['size'] = effect_size

    effect_generate = []
    for max_prob_generate in range(0,r_max_prob_generate, 3):
        effect_generate.append([max_prob_generate, calculate_noise_generate(gt_bb_list, max_prob_generate)])
    effect_noise['gen'] = effect_generate

    effect_delete = []
    for max_prob_delete in range(0,r_max_prob_delete, 3):
        effect_delete.append([max_prob_delete, calculate_noise_delete(gt_bb_list, max_prob_delete)])
    effect_noise['del'] = effect_delete


    return effect_noise
