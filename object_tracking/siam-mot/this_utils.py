import math

import numpy as np

__all__ = [
    'anchor_generator',
    'box_decode',
    'remove_small_boxes',
    'box_nms',
]


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _make_anchor():
    aspect_ratios = (0.5, 1.0, 2.0)
    anchor_strides = (4, 8, 16, 32, 64)
    anchor_sizes = (32, 64, 128, 256, 512)

    cell_anchors = [
        _generate_anchors(
            stride,
            np.array((size,), dtype=np.float) / stride,
            np.array(aspect_ratios, dtype=np.float),
        )
        for stride, size in zip(anchor_strides, anchor_sizes)
    ]

    return cell_anchors


def _grid_anchors(grid_sizes):
    strides = (4, 8, 16, 32, 64)
    cell_anchors = _make_anchor()

    anchors = []
    for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors):
        grid_height, grid_width = size
        shifts_x = np.arange(
            0, grid_width * stride, step=stride, dtype=np.float32
        )
        shifts_y = np.arange(
            0, grid_height * stride, step=stride, dtype=np.float32
        )
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

        anchors.append(
            (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
        )

    return anchors


def anchor_generator(grid_sizes):
    anchors = [
        _grid_anchors(grid_sizes)
    ]
    return anchors


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


def remove_small_boxes(bboxes, min_size):
    """
    Only keep boxes with both sides >= min_size
    """

    x1, y1, x2, y2 = np.split(bboxes, 4, axis=1)
    ws, hs = x2 - x1 + 1, y2 - y1 + 1

    keep = np.nonzero(
        (ws >= min_size) & (hs >= min_size)
    )[0]

    return bboxes[keep]


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


def box_nms(bboxes, scores, nms_thresh, max_proposals=-1):
    """
    Performs non-maximum suppression on a bboxes, with scores
    """
    keep = _box_nms(bboxes, scores, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    bboxes = bboxes[keep]

    return bboxes
