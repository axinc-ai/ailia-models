import cv2
import sys
import ailia
import numpy as np
from typing import List

sys.path.append('../../util')
from detector_utils import letterbox_convert # noqa: E402

def iou(b1: ailia.DetectorObject, b2: ailia.DetectorObject) -> float:
    area1 = b1.w * b1.h
    area2 = b2.w * b2.h
    x1 = max(b1.x, b2.x)
    y1 = max(b1.y, b2.y)
    x2 = min(b1.x + b1.w, b2.x + b2.w)
    y2 = min(b1.y + b1.h, b2.y + b2.h)

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    intersec = w * h
    union = area1 + area2 - intersec

    return intersec / union

def nms(boxes: List[ailia.DetectorObject], nms_thr: float) -> List[ailia.DetectorObject]:
    """Single class NMS implemented in Numpy."""
    keep = []
    while len(boxes) > 0:
        box = boxes.pop(0)
        keep.append(box)
        boxes = [other for other in boxes if iou(box, other) <= nms_thr]
    
    return keep

def preprocess(img: cv2.Mat, det_shape: tuple) -> cv2.Mat:
    img = letterbox_convert(img, det_shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def postprocess(img: cv2.Mat, res: np.ndarray) -> List[ailia.DetectorObject]:
    detections = []
    for out in res[0][0][0]:
        width = img.shape[1]
        height = img.shape[0]

        left = int(out[3] * width)
        top = int(out[4] * height)
        right = int(out[5] * width)
        bottom = int(out[6] * height)
        d = ailia.DetectorObject(
            category=out[1],
            prob=out[2],
            x=left,
            y=top,
            w=right - left,
            h=bottom - top,
        )
        detections.append(d)

    return nms(detections, 0.4)

def reverse_letterbox(detections: List[ailia.DetectorObject], raw_img_shape: tuple, letter_img_shape: tuple) -> List[ailia.DetectorObject]:
    rh = raw_img_shape[0]
    rw = raw_img_shape[1]
    lh = letter_img_shape[0]
    lw = letter_img_shape[1]
    
    scale = np.min((lh / rh, lw / rw))
    pad = (np.array(letter_img_shape[0:2]) - np.array(raw_img_shape[0:2]) * scale) // 2

    new_detections = []
    for d in detections:
        r = ailia.DetectorObject(
            category = d.category,
            prob = d.prob,
            x = (d.x - pad[1]) / scale,
            y = (d.y - pad[0]) / scale,
            w = d.w / scale,
            h = d.h / scale,
        )
        new_detections.append(r)
    return new_detections

