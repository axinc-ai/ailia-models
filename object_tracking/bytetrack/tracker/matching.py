import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

# from cython_bbox import bbox_overlaps

from . import kalman_filter


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return np.array(batch_shape + (rows,))
        else:
            return np.array(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = enclosed_lt = np.max(
            np.stack([
                bboxes1[..., :2],
                bboxes2[..., :2]
            ]), axis=0)  # [B, rows, 2]
        rb = enclosed_rb = np.min(
            np.stack([
                bboxes1[..., 2:],
                bboxes2[..., 2:]
            ]), axis=0)  # [B, rows, 2]

        wh = np.clip((rb - lt), 0, None)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        lt = enclosed_lt = np.max(
            np.stack([
                np.repeat(bboxes1[..., :, None, :2], bboxes2.shape[-2], axis=-2),
                np.repeat(bboxes2[..., None, :, :2], bboxes1.shape[-2], axis=-3)
            ]), axis=0)  # [B, rows, cols, 2]
        rb = enclosed_rb = np.min(
            np.stack([
                np.repeat(bboxes1[..., :, None, 2:], bboxes2.shape[-2], axis=-2),
                np.repeat(bboxes2[..., None, :, 2:], bboxes1.shape[-2], axis=-3)
            ]), axis=0)  # [B, rows, cols, 2]

        wh = np.clip((rb - lt), 0, None)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]

    union = np.where(union > eps, union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious

    # calculate gious
    enclose_wh = np.clip(enclosed_rb - enclosed_lt, 0, None)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = np.where(enclose_area > eps, enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    atlbrs = np.asarray(atlbrs)
    btlbrs = np.asarray(btlbrs)
    ious = bbox_overlaps(atlbrs, btlbrs)

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
