import numpy as np
import cv2


def bbox_based_nms(dets, thresh):
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

    return [dets[x] for x in keep]


def preprocess(image, target_shape):
    image = image / 255.0
    image = cv2.resize(image, (target_shape[1], target_shape[0]))
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]  # add one more axis to fit ailia
    return image


def pool2d(A, kernel_size, padding=1, stride=1, pool_mode='max'):
    # padding
    A = np.pad(A, padding, mode='constant')

    # window view
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = np.lib.stride_tricks.as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride*A.strides[0], stride*A.strides[1]) + A.strides,
    )
    A_w = A_w.reshape(-1, *kernel_size)
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'mean':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def topk(scores, k=40):
    cat, _, width = scores.shape

    scores = scores.reshape((cat, -1))
    topk_inds = np.argpartition(scores, -k, axis=1)[:, -k:]
    topk_scores = scores[np.arange(scores.shape[0])[:, None], topk_inds]

    topk_ys = (topk_inds / width).astype(np.int32).astype(float)
    topk_xs = (topk_inds % width).astype(np.int32).astype(float)

    topk_scores = topk_scores.reshape((-1))
    topk_ind = np.argpartition(topk_scores, -k)[-k:]
    topk_score = topk_scores[topk_ind]
    topk_classes = (topk_ind / k).astype(np.int32)
    topk_inds = gather_feat(
        topk_inds.reshape((-1, 1)), topk_ind).reshape((k))
    topk_ys = gather_feat(topk_ys.reshape((-1, 1)), topk_ind).reshape((k))
    topk_xs = gather_feat(topk_xs.reshape((-1, 1)), topk_ind).reshape((k))

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def non_maximum_suppresion(hm, kernel=3, stride=1):
    pad = (kernel - 1) // 2
    hmax = [pool2d(channel, kernel, pad, stride, 'max') for channel in hm]
    keep = (hmax == hm)
    return hm * keep


def gather_feat(feat, ind):
    dim = feat.shape[1]
    ind = np.expand_dims(ind, axis=1)
    ind = np.repeat(ind, dim, axis=1)
    feat = feat[ind, np.arange(feat.shape[1])]
    return feat


def scale_bboxes(coords, original_size, image_size):
    original_height, original_width = original_size[0], original_size[1]
    height, width = image_size[1], image_size[0]
    xmin = coords[0] / original_width * width
    ymin = coords[1] / original_height * height
    xmax = coords[2] / original_width * width
    ymax = coords[3] / original_height * height
    return [xmin, ymin, xmax, ymax, coords[4], coords[5]]


def postprocess(raw_output, image_size, k=40, threshold=0.3, iou=0.45):
    hm, reg, wh = raw_output
    hm = hm = np.exp(hm)/(1 + np.exp(hm))
    height, width = hm.shape[1:3]

    # apply nms to eliminate clusters
    hm = non_maximum_suppresion(hm)

    # extract topk
    scores, inds, classes, ys, xs = topk(hm, k=k)

    # transpose and gather feat
    reg = np.transpose(reg, (1, 2, 0))
    reg = reg.reshape((-1, reg.shape[2]))
    reg = gather_feat(reg, inds)

    reg = reg.reshape((k, 2))
    xs = xs.reshape((k, 1)) + reg[:, 0:1]
    ys = ys.reshape((k, 1)) + reg[:, 1:2]

    wh = np.transpose(wh, (1, 2, 0))
    wh = wh.reshape((-1, wh.shape[2]))
    wh = gather_feat(wh, inds)

    wh = wh.reshape((k, 2))

    classes = classes.reshape((k, 1))
    scores = scores.reshape((k, 1))
    bboxes = np.concatenate((
        xs - wh[..., 0:1] / 2,
        ys - wh[..., 1:2] / 2,
        xs + wh[..., 0:1] / 2,
        ys + wh[..., 1:2] / 2,
    ), axis=1)

    # concatenate classes, scores and bounding boxes in a single array
    detections = np.concatenate((bboxes, scores, classes), axis=1)

    filtered_detections = []
    for j in range(0, len(classes)):
        current_class = detections[np.logical_and(
            detections[..., 5] == j, detections[..., 4] >= threshold)]
        filtered_detections.extend(bbox_based_nms(current_class, iou))

    if len(filtered_detections) == 0:
        return []
    return np.apply_along_axis(
        scale_bboxes, 1, filtered_detections, (height, width), image_size
    )
