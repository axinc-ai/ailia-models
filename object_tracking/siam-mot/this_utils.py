from collections import namedtuple
import math

import numpy as np

__all__ = [
    'BBox',
    'sigmoid',
    'box_decode',
    'boxes_nms',
    'boxes_filter',
    'boxes_cat',
    'filter_results',
]

BBox = namedtuple('BBox', [
    'bbox',
    'scores',
    'ids',
    'labels',
])
BBox.__new__.__defaults__ = (None, None, None, None)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def box_decode(rel_codes, boxes, weights):
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.
    """

    boxes = boxes.astype(rel_codes.dtype)

    TO_REMOVE = 1
    widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
    heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = rel_codes[:, 0::4] / wx
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    # Prevent sending too large values into exp()
    bbox_xform_clip = math.log(1000. / 16)
    dw = np.clip(dw, None, bbox_xform_clip)
    dh = np.clip(dh, None, bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = np.exp(dw) * widths[:, None]
    pred_h = np.exp(dh) * heights[:, None]

    pred_boxes = np.zeros_like(rel_codes)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def _box_nms(dets, scores, threshold):
    x1, y1, x2, y2 = np.split(dets, 4, axis=1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]

    n = dets.shape[0]
    suppressed = np.zeros(n)

    for _i in range(n):
        i = order[_i]
        if suppressed[i] == 1:
            continue

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, n):
            j = order[_j]
            if suppressed[j] == 1:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr[0] >= threshold:
                suppressed[j] = 1

    return np.nonzero(suppressed == 0)[0]


def boxes_nms(boxes, nms_thresh, max_proposals=-1):
    """
    Performs non-maximum suppression on a bboxes, with scores
    """

    bbox = boxes.bbox
    scores = boxes.scores

    keep = _box_nms(bbox, scores, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]

    return boxes_filter(boxes, keep)


def boxes_filter(boxes, keep):
    d = dict(
        bbox=boxes.bbox[keep],
        scores=None,
        ids=None,
        labels=None,
    )
    for key in ['scores', 'ids', 'labels']:
        data = getattr(boxes, key)
        if data is not None:
            data = data[keep]
            d[key] = data

    new_boxes = BBox(**d)

    return new_boxes


def boxes_cat(boxlist):
    def _cat(xx, axis=0):
        if len(xx) == 1:
            return xx[0]
        return np.concatenate(xx, axis=axis)

    d = dict(
        bbox=_cat([b.bbox for b in boxlist], axis=0),
        scores=None,
        ids=None,
        labels=None,
    )
    for key in ['scores', 'ids', 'labels']:
        a = [getattr(b, key) for b in boxlist]
        a = [d for d in a if d is not None]
        if a:
            data = _cat(a, axis=0)
            d[key] = data

    cat_boxes = BBox(**d)

    return cat_boxes


def filter_results(bboxes, num_classes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).
    """
    # unwrap the boxlist to avoid additional overhead.
    # if we had multi-class NMS, we could perform this directly on the boxlist
    score_thresh = 0.05

    boxes = bboxes.bbox.reshape(-1, num_classes * 4)
    scores = bboxes.scores.reshape(-1, num_classes)
    ids = bboxes.ids

    result = [None for _ in range(1, num_classes)]

    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    inds_all = scores > score_thresh
    for j in range(1, num_classes):
        inds = inds_all[:, j].nonzero()[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4: (j + 1) * 4]
        ids_j = ids[inds]

        det_idx = ids_j < 0
        det_boxlist = BBox(
            bbox=boxes_j[det_idx, :],
            scores=scores_j[det_idx],
            ids=ids_j[det_idx]
        )
        det_boxlist = boxes_nms(det_boxlist, nms_thresh=0.5)

        track_idx = ids_j >= 0
        # track_box is available
        if np.any(track_idx > 0):
            track_boxlist = BBox(
                bbox=boxes_j[track_idx, :],
                scores=scores_j[track_idx],
                ids=ids_j[track_idx],
            )
            det_boxlist = boxes_cat([det_boxlist, track_boxlist])

        num_labels = len(det_boxlist.bbox)
        det_boxlist = BBox(
            bbox=det_boxlist.bbox,
            scores=det_boxlist.scores,
            ids=det_boxlist.ids,
            labels=np.full((num_labels,), j)
        )
        result[j - 1] = det_boxlist

    result = boxes_cat(result)

    return result
