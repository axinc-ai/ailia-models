import numpy as np
import cv2

from nms_utils import nms_boxes

import face_align

onnx = False


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


def det_preprocess(img):
    det_wh = (640, 640)

    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(det_wh[1]) / det_wh[0]
    if im_ratio > model_ratio:
        new_height = det_wh[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = det_wh[0]
        new_height = int(new_width * im_ratio)

    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    img = np.zeros((det_wh[1], det_wh[0], 3), dtype=np.uint8)
    img[:new_height, :new_width, :] = resized_img

    img = cv2.dnn.blobFromImage(img, 1.0 / 128, (det_wh[1], det_wh[0]), (127.5, 127.5, 127.5), swapRB=True)

    return img, det_scale


def detect_face(img, net_iface, nms_threshold):
    det_thresh = 0.6
    img, det_scale = det_preprocess(img)

    # feedforward
    if not onnx:
        net_outs = net_iface.predict([img])
    else:
        net_outs = net_iface.run(None, {'input.1': img})

    input_height, input_width = img.shape[2:]
    fmc = 3
    feat_stride_fpn = [8, 16, 32]
    num_anchors = 2
    center_cache = {}

    scores_list = []
    bboxes_list = []
    kpss_list = []
    for idx, stride in enumerate(feat_stride_fpn):
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]
        bbox_preds = bbox_preds * stride
        kps_preds = net_outs[idx + fmc * 2] * stride
        height = input_height // stride
        width = input_width // stride
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            if len(center_cache) < 100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= det_thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        kpss = distance2kps(anchor_centers, kps_preds)
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        pos_kpss = kpss[pos_inds]
        kpss_list.append(pos_kpss)

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    bboxes = np.vstack(bboxes_list) / det_scale
    kpss = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms_boxes(pre_det[:, :4], pre_det[:, 4], nms_threshold)
    det = pre_det[keep, :]
    kpss = kpss[order, :, :]
    kpss = kpss[keep, :, :]

    return det, kpss


def get_kps(img, net_iface, nms_threshold=0.4):
    """
    Crop face from image and resize
    """

    bboxes, kpss = detect_face(img, net_iface, nms_threshold)

    if bboxes.shape[0] == 0:
        return None

    kps_list = []
    for i in range(bboxes.shape[0]):
        kps = None
        if kpss is not None:
            kps = kpss[i]
        kps_list.append(kps)

    return kps_list


def crop_face(img, net_iface, crop_size, nms_threshold=0.4):
    """
    Crop face from image and resize
    """

    kps = get_kps(img, net_iface, nms_threshold)

    if kps is None:
        return None

    M, _ = face_align.estimate_norm(kps[0], crop_size, mode='None')
    align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

    return align_img
