from typing import List

import numpy as np

from face import Face

INPUT_SIZE = (320, 320)
DET_THRESH = 0.5
NMS_THRESH = 0.4
INPUT_STD = 128.0
INPUT_MEAN = 127.5
USE_KPS = True
FMC = 3
NUM_ANCHORS = 2
FEAT_STRIDE_FPN = [8, 16, 32]


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def forward(net_outs, use_kps: bool, thresh=DET_THRESH):
    center_cache = {}
    scores_list = []
    bboxes_list = []
    kpss_list = []
    input_height = 320
    input_width = 320

    for idx, stride in enumerate(FEAT_STRIDE_FPN):
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + FMC]
        bbox_preds = bbox_preds * stride
        if use_kps:
            kps_preds = net_outs[idx + FMC * 2] * stride

        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)

        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32
            )

            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if NUM_ANCHORS > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * NUM_ANCHORS, axis=1
                ).reshape((-1, 2))
            if len(center_cache) < 100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

    return scores_list, bboxes_list, kpss_list


def nms(dets, thresh=NMS_THRESH):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def detect(net_outs, use_kps: bool, det_scale):
    scores_list, bboxes_list, kpss_list = forward(net_outs, use_kps)

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale

    if use_kps:
        kpss = np.vstack(kpss_list) / det_scale

    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]

    if use_kps:
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]
    else:
        kpss = None

    return det, kpss


def get_detection(net_outs, use_kps: bool, det_scale: float) -> List[Face]:
    bboxes, kpss = detect(net_outs["detection"], use_kps, det_scale)

    if bboxes.shape[0] == 0:
        return []

    ret = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)

        ret.append(face)

    return ret
