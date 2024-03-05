import os
from itertools import product as product
from math import ceil
import numpy as np

cfg_mnet = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}


class RetinaFaceOnnx:
    def __init__(
        self,
        model
    ):
        self.model = model

    def __call__(self, images):
        output = batch_detect(self.model, [images])[0]
        return output

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase="train"):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1] for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0] for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        #output = torch.Tensor(anchors).view(-1, 4)
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def batch_detect(net, images):
    confidence_threshold = 0.02
    cfg = cfg_mnet
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    resize = 1
    img = np.float32(images)
    mean = np.array([[[[104, 117, 123]]]], dtype=img.dtype)
    img -= mean
    img = img.transpose(0, 3, 1, 2)
    batch_size, _, im_height, im_width, = img.shape
    scale = np.array(
        [im_width, im_height, im_width, im_height],
        dtype=img.dtype
    )
    loc, conf, landms = net.run(img)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    prior_data = priorbox.forward()
    scale1 = np.array(
        [
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
        ],
        dtype=img.dtype
    )

    all_dets = [
        post_process(
            loc_i,
            conf_i,
            landms_i,
            prior_data,
            cfg,
            scale,
            scale1,
            resize,
            confidence_threshold,
            top_k,
            nms_threshold,
            keep_top_k,
        )
        for loc_i, conf_i, landms_i in zip(loc, conf, landms)
    ]

    return all_dets

def post_process(
    loc,
    conf,
    landms,
    prior_data,
    cfg,
    scale,
    scale1,
    resize,
    confidence_threshold,
    top_k,
    nms_threshold,
    keep_top_k,
):

    boxes = decode(loc, prior_data, cfg["variance"])
    boxes = boxes * scale / resize
    boxes = boxes
    scores = conf[:, 1]

    landms_copy = decode_landm(landms, prior_data, cfg["variance"])

    landms_copy = landms_copy * scale1 / resize
    landms_copy = landms_copy

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms_copy = landms_copy[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms_copy = landms_copy[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms_copy = landms_copy[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms_copy = landms_copy[:keep_top_k, :]

    dets = np.concatenate((dets, landms_copy), axis=1)
    # show image
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    dets = [parse_det(x) for x in dets]

    return dets

def decode(loc, priors, variances):
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    landms = np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )
    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
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

def parse_det(det):
    landmarks = det[5:].reshape(5, 2)
    box = det[:4]
    score = det[4]
    return box, landmarks, score