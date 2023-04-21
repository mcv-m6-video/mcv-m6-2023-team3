import os
from collections import defaultdict, OrderedDict

import numpy as np


from utils import Detection


def parse_annotations(path):
    with open(path) as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        data = line.split(',')
        annotations.append(Detection(
            frame=int(data[0])-1,
            id=int(data[1]),
            label='car',
            xtl=float(data[2]),
            ytl=float(data[3]),
            xbr=float(data[2])+float(data[4]),
            ybr=float(data[3])+float(data[5]),
        ))

    return annotations

def group_by_frame(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.frame].append(det)
    return OrderedDict(sorted(grouped.items()))


class DataAnotations:

    def __init__(self, path):
        self.annotations = parse_annotations(path)
        self.classes = np.unique([det.label for det in self.annotations])

    def get_annotations(self, classes=None, noise_params=None, do_group_by_frame=True, only_not_parked=False):
        if classes is None:
            classes = self.classes

        detections = []
        for det in self.annotations:
            if det.label in classes:  # filter by class
                if only_not_parked and det.parked:
                    continue
                detections.append(det)

        if do_group_by_frame:
            detections = group_by_frame(detections)

        return detections
