from itertools import product as product
import numpy as np
import cv2


class PriorBox(object):
    def __init__(self, box_dimension=None, image_size=None):
        super(PriorBox, self).__init__()
        self.variance = [0.1, 0.2]
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.aspect_ratios = [[1], [1], [1]]
        self.image_size = image_size
        self.feature_maps = box_dimension
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1]
                                    for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0]
                                    for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            mean += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1]
                                    for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0]
                                    for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            mean += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        mean += [cx, cy, s_kx, s_ky]
        output = np.array(mean).reshape(-1, 4)
        return output


RESIZE = 1


def pre_process(to_show):
    img = np.float32(to_show)
    img = cv2.RESIZE(
        img, None, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_LINEAR
    )
    scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img, scale


def get_detection_dimension(out):
    detection_dimension = []
    for i in range(2, len(out)):
        detection_dimension.append(int(out[i]))
    detection_dimension = np.array(detection_dimension).reshape(3, 2).tolist()
    return detection_dimension


def decode(loc, priors, variances):
    boxes = np.concatenate([
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    ], 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(dets, thresh):
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


def post_process(out, img, scale, THRESHOLD, IOU):
    detection_dimension = get_detection_dimension(out)
    priorbox = PriorBox(detection_dimension, (img.shape[2], img.shape[3]))
    priors = priorbox.forward()
    loc, conf = out[0], out[1]
    boxes = decode(loc.squeeze(0), priors, [0.1, 0.2])
    boxes = boxes * scale / RESIZE
    scores = conf[:, 1]

    # ignore low scores
    inds = np.where(scores > THRESHOLD)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack(
        (boxes, scores[:, np.newaxis])
    ).astype(np.float32, copy=False)
    keep = nms(dets, IOU)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:750, :]
    return dets
