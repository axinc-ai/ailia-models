from typing import Union

import cv2
import numpy as np

__all__ = [
    'invert_affine',
    'aspectaware_resize_padding',
    'bbox_transform',
    'clip_boxes',
]


def invert_affine(metas, preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
            preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
            preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)

    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h


def bbox_transform(anchors, regression):
    """
    decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

    Args:
        anchors: [batchsize, boxes, (y1, x1, y2, x2)]
        regression: [batchsize, boxes, (dy, dx, dh, dw)]

    Returns:

    """
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = np.exp(regression[..., 3]) * wa
    h = np.exp(regression[..., 2]) * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return np.stack([xmin, ymin, xmax, ymax], axis=2)


def clip_boxes(boxes, img):
    _, _, height, width = img.shape

    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, None)
    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, None)
    boxes[:, :, 2] = np.clip(boxes[:, :, 2], None, width - 1)
    boxes[:, :, 3] = np.clip(boxes[:, :, 3], None, height - 1)

    return boxes
