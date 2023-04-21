import numpy as np
# from https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

class Track():
    def __init__(self, id, detections):
        self.id = id
        self.detections = detections
        self.terminated = False
        color = (list(np.random.choice(range(256), size=3))) 
        self.color =(int(color[0]), int(color[1]), int(color[2]))

    def add_detection(self, detection):
        self.detections.append(detection)

class Detection():

    def __init__(self, frame, id, label, xtl, ytl, xbr, ybr):
        self.frame = frame
        self.id = id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.bbox = [self.xtl, self.ytl, self.xbr, self.ybr]
        self.width = abs(self.xbr - self.xtl)
        self.height = abs(self.ybr - self.ytl)
        self.area = self.width * self.height
        self.center = (int((self.xtl + self.xbr) / 2), int((self.ytl + self.ybr) / 2))

#from https://github.com/mcv-m6-video/mcv-m6-2020-team2
def refine_bbox(last_detections, new_detection, k=0.5):
    # No refinement for the first two frames
    if len(last_detections) < 2:
        return new_detection

    # Predict coordinates of new detection from last two detections
    pred_detection_xtl = 2 * last_detections[1].xtl - last_detections[0].xtl
    pred_detection_ytl = 2 * last_detections[1].ytl - last_detections[0].ytl
    pred_detection_xbr = 2 * last_detections[1].xbr - last_detections[0].xbr
    pred_detection_ybr = 2 * last_detections[1].ybr - last_detections[0].ybr

    # Compute average of predicted coordinates and detected coordinates
    refined_xtl = new_detection.xtl * k + pred_detection_xtl * (1 - k)
    refined_ytl = new_detection.ytl * k + pred_detection_ytl * (1 - k)
    refined_xbr = new_detection.xbr * k + pred_detection_xbr * (1 - k)
    refined_ybr = new_detection.ybr * k + pred_detection_ybr * (1 - k)

    # Get refined detection
    refined_detection = Detection(frame=new_detection.frame,
                                  id=new_detection.id,
                                  label=new_detection.label,
                                  xtl=refined_xtl,
                                  ytl=refined_ytl,
                                  xbr=refined_xbr,
                                  ybr=refined_ybr)

    return refined_detection
