from collections import namedtuple

import numpy as np
import cv2

from objectron.dataset import graphics

__all__ = [
    'draw_kp',
    'ssd_anchors',
    'decode_boxes',
    'non_max_suppression',
]


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    else:
        return min_scale + \
               (max_scale - min_scale) * 1 * stride_index / (num_strides - 1)


def ssd_anchors():
    num_layers = 6
    min_scale = 0.2
    max_scale = 0.95
    input_size_height = 300
    input_size_width = 300
    anchor_offset_x = 0.5
    anchor_offset_y = 0.5
    strides = [16, 32, 64, 128, 256, 512]
    aspect_ratios = [
        1.0, 2.0, 0.5, 3.0, 0.333
    ]
    reduce_boxes_in_lowest_layer = True
    interpolated_scale_aspect_ratio = 1.0

    strides_size = len(strides)
    aspect_ratios_size = len(aspect_ratios)

    anchor_t = namedtuple('anchor', ['x_center', 'y_center', 'w', 'h'])
    anchors = []
    layer_id = 0
    while layer_id < num_layers:
        anchor_height = []
        anchor_width = []
        _aspect_ratios = []
        _scales = []

        last_same_stride_layer = layer_id
        while last_same_stride_layer < strides_size \
                and strides[last_same_stride_layer] == strides[layer_id]:
            scale = calculate_scale(min_scale, max_scale, last_same_stride_layer, strides_size)

            if last_same_stride_layer == 0 and reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                _aspect_ratios.append(1.0)
                _aspect_ratios.append(2.0)
                _aspect_ratios.append(0.5)
                _scales.append(0.1)
                _scales.append(scale)
                _scales.append(scale)
            else:
                for i in range(aspect_ratios_size):
                    _aspect_ratios.append(aspect_ratios[i])
                    _scales.append(scale)
                if 0.0 < interpolated_scale_aspect_ratio:
                    scale_next = 1.0 \
                        if last_same_stride_layer == strides_size - 1 \
                        else calculate_scale(
                        min_scale, max_scale,
                        last_same_stride_layer + 1,
                        strides_size)
                    _scales.append((scale * scale_next) ** 0.5)
                    _aspect_ratios.append(interpolated_scale_aspect_ratio)

            last_same_stride_layer += 1

        for i in range(len(_aspect_ratios)):
            ratio_sqrts = (_aspect_ratios[i]) ** 0.5
            anchor_height.append(_scales[i] / ratio_sqrts)
            anchor_width.append(_scales[i] * ratio_sqrts)

        stride = strides[layer_id]
        feature_map_height = int(np.ceil(1.0 * input_size_height / stride))
        feature_map_width = int(np.ceil(1.0 * input_size_width / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + anchor_offset_x) * 1.0 / feature_map_width
                    y_center = (y + anchor_offset_y) * 1.0 / feature_map_height
                    w = anchor_width[anchor_id]
                    h = anchor_height[anchor_id]
                    new_anchor = anchor_t(x_center, y_center, w, h)
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    return anchors


def decode_boxes(boxes, anchors):
    x_scale = 10.0
    y_scale = 10.0
    h_scale = 5.0
    w_scale = 5.0
    apply_exponential_on_box_size = True

    detected_boxes = []
    for i, box in enumerate(boxes):
        y_center = box[0]
        x_center = box[1]
        h = box[2]
        w = box[3]

        x_center = x_center / x_scale * anchors[i].w + anchors[i].x_center
        y_center = y_center / y_scale * anchors[i].h + anchors[i].y_center

        if apply_exponential_on_box_size:
            h = np.exp(h / h_scale) * anchors[i].h
            w = np.exp(w / w_scale) * anchors[i].w
        else:
            h = h / h_scale * anchors[i].h
            w = w / w_scale * anchors[i].w

        ymin = y_center - h / 2.
        xmin = x_center - w / 2.
        ymax = y_center + h / 2.
        xmax = x_center + w / 2.
        detected_boxes.append([xmin, ymin, xmax, ymax])

    return detected_boxes


def overlap_similarity(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    interArea = abs(xB - xA) * abs(yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs(box_a[2] - box_a[0]) * abs(box_a[3] - box_a[1])
    boxBArea = abs(box_b[2] - box_b[0]) * abs(box_b[3] - box_b[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    similarity = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return similarity


def non_max_suppression(scores, boxes, classes, max_num_detections=100):
    min_suppression_threshold = 0.5

    retained = []
    for i in reversed(np.argsort(scores)):
        suppressed = False
        box_a = boxes[i]
        for j in retained:
            box_b = boxes[j]
            similarity = overlap_similarity(box_a, box_b)
            if similarity > min_suppression_threshold:
                suppressed = True
                break
        if suppressed is False:
            retained.append(i)
            if len(retained) >= max_num_detections:
                break

    scores = [scores[i] for i in retained]
    boxes = [boxes[i] for i in retained]
    classes = [classes[i] for i in retained]

    return scores, boxes, classes


def normalize(image_shape, unnormalized_keypoints):
    ''' normalize keypoints to image coordinates '''
    assert len(image_shape) in [2, 3]
    if len(image_shape) == 3:
        h, w, _ = image_shape
    else:
        h, w = image_shape

    keypoints = unnormalized_keypoints / np.asarray([w, h], np.float32)
    return keypoints


def draw_kp(
        img, keypoints, normalized=True, num_keypoints=9, label=None):
    '''
    img: numpy three dimensional array
    keypoints: array like with shape [9,2]
    name: path to save
    '''
    img_copy = img.copy()
    # if image transposed
    if img_copy.shape[0] == 3:
        img_copy = np.transpose(img_copy, (1, 2, 0))
    # expand dim with zeros, needed for drawing function API
    expanded_kp = np.zeros((num_keypoints, 3))
    keypoints = keypoints if normalized else normalize(img_copy.shape, keypoints)
    expanded_kp[:, :2] = keypoints
    graphics.draw_annotation_on_image(img_copy, expanded_kp, [num_keypoints])
    # put class label if given
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_copy, str(label), (10, 180), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img_copy
