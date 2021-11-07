from collections import namedtuple

import numpy as np

__all__ = [
    'detection_output'
]


def intersection_area(a, b):
    if a.xmin > b.xmax or a.xmax < b.xmin or a.ymin > b.ymax or a.ymax < b.ymin:
        return 0

    inter_width = min(a.xmax, b.xmax) - max(a.xmin, b.xmin);
    inter_height = min(a.ymax, b.ymax) - max(a.ymin, b.ymin);

    return inter_width * inter_height;


def nms_sorted_bboxes(bboxes, nms_threshold):
    n = len(bboxes)
    picked = []

    areas = []
    for i in range(n):
        r = bboxes[i]
        width = r.xmax - r.xmin
        height = r.ymax - r.ymin
        areas.append(width * height)

    for i in range(n):
        a = bboxes[i]

        keep = True
        for j in range(len(picked)):
            b = bboxes[picked[j]]

            # intersection over union
            inter_area = intersection_area(a, b)
            union_area = areas[i] + areas[picked[j]] - inter_area

            if inter_area / union_area > nms_threshold:
                keep = False

        if keep:
            picked.append(i)

    return picked


def detection_output(conf, loc):
    num_class = 3
    nms_top_k = 300
    keep_top_k = 200
    confidence_threshold = 0.01
    nms_threshold = 0.45

    priorbox = np.load("anchor.npy")

    confidence = conf.reshape(-1, 3)  # 1126, 3
    location = loc.reshape(-1, 4)  # 1126, 4
    variance = priorbox[0][1].reshape(-1, 4)  # 1126,4
    priorbox = priorbox[0][0].reshape(-1, 4)  # 1126, 6

    num_prior = len(priorbox)
    bboxes = np.zeros((num_prior, 4))
    for i in range(num_prior):
        # if score of background class is larger than confidence threshold
        score = confidence[i]
        if score[0] >= 1.0 - confidence_threshold:
            continue

        loc = location[i]
        pb = priorbox[i]
        var = variance[i]

        # CENTER_SIZE
        pb_w = pb[2] - pb[0]
        pb_h = pb[3] - pb[1]
        pb_cx = (pb[0] + pb[2]) / 2
        pb_cy = (pb[1] + pb[3]) / 2

        bbox_cx = var[0] * loc[0] * pb_w + pb_cx
        bbox_cy = var[1] * loc[1] * pb_h + pb_cy
        bbox_w = np.exp(var[2] * loc[2]) * pb_w
        bbox_h = np.exp(var[3] * loc[3]) * pb_h
        bboxes[i, :] = [
            bbox_cx - bbox_w / 2,
            bbox_cy - bbox_h / 2,
            bbox_cx + bbox_w / 2,
            bbox_cy + bbox_h / 2,
        ]

    # sort and nms for each class
    all_class_bbox_rects = [[]]
    all_class_bbox_scores = [[]]

    # start from 1 to ignore background class
    for i in range(1, num_class):
        # filter by confidence_threshold
        class_bbox_rects = []
        class_bbox_scores = []

        for j in range(num_prior):
            # prob data layout
            # num_class x num_prior
            score = confidence[j, i]

            if score > confidence_threshold:
                bbox = bboxes[j]
                class_bbox_rects.append((bbox[0], bbox[1], bbox[2], bbox[3], i))
                class_bbox_scores.append(score)

        idx = np.argsort(class_bbox_scores)[::-1]
        class_bbox_rects = np.asarray(class_bbox_rects)[idx]
        class_bbox_scores = np.asarray(class_bbox_scores)[idx]

        class_bbox_rects = class_bbox_rects[:nms_top_k]
        class_bbox_scores = class_bbox_scores[:nms_top_k]

        BBoxRect = namedtuple('BBoxRect', ['xmin', 'ymin', 'xmax', 'ymax', 'label'])
        class_bbox_rects = [
            BBoxRect(*b) for b in class_bbox_rects
        ]

        # apply nms
        picked = nms_sorted_bboxes(class_bbox_rects, nms_threshold)

        # select
        all_class_bbox_rects.append([])
        all_class_bbox_scores.append([])
        for z in picked:
            all_class_bbox_rects[i].append(class_bbox_rects[z])
            all_class_bbox_scores[i].append(class_bbox_scores[z])

    # gather all class
    bbox_rects = []
    bbox_scores = []
    for i in range(1, num_class):
        class_bbox_rects = all_class_bbox_rects[i]
        class_bbox_scores = all_class_bbox_scores[i]
        bbox_rects.extend(class_bbox_rects)
        bbox_scores.extend(class_bbox_scores)

    idx = np.argsort(bbox_scores)[::-1]
    bbox_rects = [bbox_rects[i] for i in idx]
    bbox_scores = [bbox_scores[i] for i in idx]

    bbox_rects = bbox_rects[:keep_top_k]
    bbox_scores = bbox_scores[:keep_top_k]

    detections = []
    for i in range(len(bbox_rects)):
        r = bbox_rects[i]
        score = bbox_scores[i]
        detections.append([
            r.label,
            score,
            r.xmin, r.ymin,
            r.xmax, r.ymax
        ])

    return detections
