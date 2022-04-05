import sys
import os
import argparse
import numpy as np
import cv2

import ailia


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area1 = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None)
    inter_area2 = np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)

    inter_area = inter_area1 * inter_area2

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    box_corner = np.zeros(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2 # cx - w/2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2 # cy - h/2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2 # cx + w/2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2 # cy + h/2
    prediction[:, :, :4] = box_corner[:, :, :4] #conf

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.shape[0]:
            continue

        class_conf = np.max(image_pred[:, 5:5 + num_classes], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=1)
        class_pred = class_pred.reshape((class_pred.shape[0],1))
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), 1)
        unique_labels = np.unique(detections[:, -1])

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]

            conf_sort_value = np.sort(detections_class[:, 4])
            conf_sort_value = conf_sort_value[::-1]

            conf_sort_index = np.argsort(detections_class[:, 4])
            conf_sort_index = conf_sort_index[::-1]

            detections_class = detections_class[conf_sort_index]

            max_detections = []
            while detections_class.shape[0]:
                expand_detections_class = np.expand_dims(detections_class[0],0)
                max_detections.append(expand_detections_class)
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(expand_detections_class, detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = np.concatenate(max_detections)
            
            output[image_i] = max_detections if output[image_i] is None else np.concatenate(
                (output[image_i], max_detections))

    return output


def post_processing(img, conf_thres, nms_thres, outputs):
    batch_detections = []

    img_size_w = img.shape[3]
    img_size_h = img.shape[2]

    batch_size = 1
    num_classes = 80

    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

    boxs = []
    a = np.array(anchors).reshape(3, -1, 2)
    anchor_grid = a.copy().reshape(3, 1, -1, 1, 1, 2)

    if len(outputs)==1:
        # yolov5 v6
        outputx = outputs[0]
    else:
        # yolov5 v1
        for index, out in enumerate(outputs):
            batch = out.shape[1]
            feature_h = out.shape[2]
            feature_w = out.shape[3]

            # Feature map corresponds to the original image zoom factor
            stride_w = int(img_size_w / feature_w)
            stride_h = int(img_size_h / feature_h)

            grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

            # cx, cy, w, h
            pred_boxes = np.zeros(out[..., :4].shape)
            pred_boxes[..., 0] = (sigmoid(out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
            pred_boxes[..., 1] = (sigmoid(out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
            pred_boxes[..., 2:4] = (sigmoid(out[..., 2:4]) * 2) ** 2 * anchor_grid[index]  # wh

            conf = sigmoid(out[..., 4])
            pred_cls = sigmoid(out[..., 5:])

            output = np.concatenate((pred_boxes.reshape(batch_size, -1, 4),
                                conf.reshape(batch_size, -1, 1),
                                pred_cls.reshape(batch_size, -1, num_classes)),
                                -1)
            boxs.append(output)

        outputx = np.concatenate(boxs, 1)

    # NMS
    batch_detections = non_max_suppression(outputx, num_classes, conf_thres=conf_thres, nms_thres=nms_thres)

    # output ailia format
    detections = batch_detections[0]
    if detections is None:
        return [[]]

    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    bboxes = []

    bboxes_batch = []
    for i, box in enumerate(boxs):
        x1, y1, x2, y2 = box
        c = int(labels[i])
        r = ailia.DetectorObject(
            category=c,
            prob=confs[i],
            x=x1/img_size_w,
            y=y1/img_size_h,
            w=(x2 - x1)/img_size_w,
            h=(y2 - y1)/img_size_h,
        )
        bboxes.append(r)
    bboxes_batch.append(bboxes)
    
    return bboxes_batch
