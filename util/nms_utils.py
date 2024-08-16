import numpy as np


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


def nms_between_categories(detections, w, h, categories=None, iou_threshold=0.25):
    # Normally darknet use per class nms
    # But some cases need between class nms
    # https://github.com/opencv/opencv/issues/17111

    # remove overwrapped detection
    det = []
    keep = []
    for idx in range(len(detections)):
        obj = detections[idx]
        is_keep = True
        for idx2 in range(len(det)):
            if not keep[idx2]:
                continue
            box_a = [w * det[idx2].x, h * det[idx2].y, w * (det[idx2].x + det[idx2].w), h * (det[idx2].y + det[idx2].h)]
            box_b = [w * obj.x, h * obj.y, w * (obj.x + obj.w), h * (obj.y + obj.h)]
            iou = bb_intersection_over_union(box_a, box_b)
            if iou >= iou_threshold and (
                    categories == None or ((det[idx2].category in categories) and (obj.category in categories))):
                if det[idx2].prob <= obj.prob:
                    keep[idx2] = False
                else:
                    is_keep = False
        det.append(obj)
        keep.append(is_keep)

    det = []
    for idx in range(len(detections)):
        if keep[idx]:
            det.append(detections[idx])

    return det


def nms_boxes(boxes, scores, iou_thres):
    # Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    keep = []
    for i, box_a in enumerate(boxes):
        is_keep = True
        for j in range(i):
            if not keep[j]:
                continue
            box_b = boxes[j]
            iou = bb_intersection_over_union(box_a, box_b)
            if iou >= iou_thres:
                if scores[i] > scores[j]:
                    keep[j] = False
                else:
                    is_keep = False
                    break

        keep.append(is_keep)

    return np.array(keep).nonzero()[0]


def batched_nms(boxes, scores, labels, iou_thres):
    a = []
    for i in np.unique(labels):
        idx = (labels == i)
        idx = np.nonzero(idx)[0]
        i = nms_boxes(boxes[idx], scores[idx], iou_thres)
        idx = idx[i]
        a.append(idx)

    keep = np.concatenate(a)
    scores = scores[keep]
    idxs = np.argsort(-scores)
    keep = keep[idxs]

    return keep


def packed_nms(boxes, scores, iou_thres):
    packed_idx = []
    remained = np.argsort(-scores)
    while 0 < len(remained):
        idx = remained
        i = idx[0]
        candidates = [i]
        remained = []
        for j in idx[1:]:
            similarity = bb_intersection_over_union(boxes[i], boxes[j])
            if similarity > iou_thres:
                candidates.append(j)
            else:
                remained.append(j)

        packed_idx.append(candidates)

    return packed_idx
