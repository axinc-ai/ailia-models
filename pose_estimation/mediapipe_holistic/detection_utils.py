from collections import namedtuple

import numpy as np

from math_utils import sigmoid
from nms_utils import packed_nms

IMAGE_SIZE = 224


def calc_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    else:
        return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0)


def get_anchor(num_layers, strides, input_height, input_width):
    opt_aspect_ratios = [1.0]
    min_scale = 0.1484375
    max_scale = 0.75
    anchor_offset_x = 0.5
    anchor_offset_y = 0.5
    interpolated_scale_aspect_ratio = 1.0

    Anchor = namedtuple('Anchor', ['x_center', 'y_center', 'w', 'h'])
    anchors = []

    layer_id = 0
    while layer_id < num_layers:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        last_same_stride_layer = layer_id
        while last_same_stride_layer < len(strides) \
                and strides[last_same_stride_layer] == strides[layer_id]:

            scale = calc_scale(min_scale, max_scale, last_same_stride_layer, len(strides))
            for aspect_ratio in opt_aspect_ratios:
                aspect_ratios.append(aspect_ratio)
                scales.append(scale)

            if last_same_stride_layer == len(strides) - 1:
                scale_next = 1.0
            else:
                scale_next = calc_scale(
                    min_scale, max_scale, last_same_stride_layer + 1, len(strides))
            scales.append((scale * scale_next) ** 0.5)
            aspect_ratios.append(interpolated_scale_aspect_ratio)

            last_same_stride_layer += 1

        for i, aspect_ratio in enumerate(aspect_ratios):
            ratio_sqrts = aspect_ratio ** 0.5
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = strides[layer_id]
        feature_map_height = int(np.ceil(1.0 * input_height / stride))
        feature_map_width = int(np.ceil(1.0 * input_width / stride))

        for y in np.arange(feature_map_height):
            for x in np.arange(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + anchor_offset_x) * 1.0 / feature_map_width
                    y_center = (y + anchor_offset_y) * 1.0 / feature_map_height
                    w = h = 1.0
                    anchors.append(Anchor(x_center, y_center, w, h))

        layer_id = last_same_stride_layer

    return anchors


def decode_boxes(raw_boxes, anchors, num_boxes, num_coords, num_keypoints, scale):
    x_scale = y_scale = scale
    w_scale = h_scale = scale

    # (num_boxes, (xmin, ymin, xmax, ymax, key1_x, key1_y, key2_x, key2_y, key3_x, key3_y, key4_x, key4_y, ...))
    boxes = np.zeros((num_boxes, num_coords))
    for i in range(num_boxes):
        x_center = raw_boxes[i][0]
        y_center = raw_boxes[i][1]
        w = raw_boxes[i][2]
        h = raw_boxes[i][3]

        x_center = x_center / x_scale * anchors[i].w + anchors[i].x_center
        y_center = y_center / y_scale * anchors[i].h + anchors[i].y_center
        h = h / h_scale * anchors[i].h
        w = w / w_scale * anchors[i].w

        ymin = y_center - h / 2.
        xmin = x_center - w / 2.
        ymax = y_center + h / 2.
        xmax = x_center + w / 2.

        boxes[i][0] = xmin
        boxes[i][1] = ymin
        boxes[i][2] = xmax
        boxes[i][3] = ymax
        for k in range(num_keypoints):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[i][offset]
            keypoint_y = raw_boxes[i][offset + 1]
            boxes[i][offset] = keypoint_x / x_scale * anchors[i].w + anchors[i].x_center
            boxes[i][offset + 1] = keypoint_y / y_scale * anchors[i].h + anchors[i].y_center

    return boxes


def weighted_nms(boxes, scores, img_size):
    min_suppression_threshold = 0.3

    px_boxes = np.zeros((len(boxes), 4))
    px_boxes[:, :4] = boxes[:, :4] * img_size

    packed_idx = packed_nms(px_boxes, scores, min_suppression_threshold)

    out_boxes = []
    out_scores = []
    for idx in packed_idx:
        total_score = np.sum(scores[idx])

        candidates = boxes[idx]
        candidates = candidates * scores[idx].reshape(-1, 1)
        weighted_detection = np.sum(candidates, axis=0) / total_score

        out_boxes.append(weighted_detection)
        out_scores.append(np.max(scores[idx]))

    if len(out_boxes) == 0:
        return [], []

    out_boxes = np.vstack(out_boxes)
    out_scores = np.array(out_scores)

    return out_boxes, out_scores


anchors = get_anchor(
    num_layers=5,
    strides=[8, 16, 32, 32, 32],
    input_height=IMAGE_SIZE,
    input_width=IMAGE_SIZE)


def pose_detection(detections, scores, pad):
    num_boxes = 2254
    num_coords = 12
    num_keypoints = 4
    img_size = IMAGE_SIZE
    boxes = decode_boxes(detections[0], anchors, num_boxes, num_coords, num_keypoints, img_size)
    scores = np.clip(scores[0, :, 0], -100, 100)
    scores = sigmoid(scores)

    min_score_thresh = 0.5
    idx = scores >= min_score_thresh
    boxes = boxes[idx]
    scores = scores[idx]

    # Performs non-max suppression to remove excessive detections.
    boxes, scores = weighted_nms(boxes, scores, img_size)

    if len(boxes) == 0:
        return [], []

    # Adjusts detection locations (already normalized to [0.f, 1.f]) on the
    # letterboxed image (after image transformation with the FIT scale mode)
    pad_h, pad_w = pad
    if 0 < pad_w:
        boxes[:, [0, 2, 4, 6, 8, 10]] = (boxes[:, [0, 2, 4, 6, 8, 10]] - pad_w) / (1.0 - pad_w * 2)
    if 0 < pad_h:
        boxes[:, [1, 3, 5, 7, 9, 11]] = (boxes[:, [1, 3, 5, 7, 9, 11]] - pad_h) / (1.0 - pad_h * 2)

    # Gets the very first detection
    box = boxes[0]
    score = scores[0]

    return box, score
