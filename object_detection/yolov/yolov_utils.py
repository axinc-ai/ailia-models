import cv2
import numpy as np
import json
import ailia

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r



class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None,t_size = 0.4):

    result = []
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        img_size_h, img_size_w = img.shape[:2]
        r = ailia.DetectorObject(
            category=cls_id,
            prob=score,
            x=x0 / img_size_w,
            y=y0 / img_size_h,
            w=(x1 - x0) / img_size_w,
            h=(y1 - y0) / img_size_h,
        )
        result.append(r)

    return result


def save_result_json(json_path, output, ratio, conf=0.5, class_names=None, t_size=0.4):
    result = []

    boxes = output[:, 0:4]
    boxes /= ratio
    scores = output[:, 4] * output[:, 5]
    cls_ids = output[:, 6]

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        result.append({
            'category': class_names[cls_id] if class_names is not None else cls_id,
            'prob': float(score),
            'box': box.tolist()
        })
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
