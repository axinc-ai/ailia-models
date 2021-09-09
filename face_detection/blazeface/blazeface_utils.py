import cv2
import numpy as np
import ailia

import sys
sys.path.append('../../util')
from detector_utils import letterbox_convert, reverse_letterbox  # noqa: E402


DEFAULT_MIN_SCORE_THRESH = 0.75


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def plot_detections(
        img, detections, with_keypoints=True, save_image_path='result.png'
):
    # color in BGR
    box_color = (0, 0, 255)
    dot_color = (0xfa, 0xce, 0x87)
    thickness = 2

    for i in range(detections.shape[0]):
        ymin = int(detections[i, 0] * img.shape[0])
        xmin = int(detections[i, 1] * img.shape[1])
        ymax = int(detections[i, 2] * img.shape[0])
        xmax = int(detections[i, 3] * img.shape[1])

        p0 = (xmin, ymin)
        p1 = (xmax, ymax)

        img = cv2.rectangle(img, p0, p1, box_color, thickness)

        if with_keypoints:
            for k in range(6):
                kp_x = int(detections[i, 4 + k*2    ] * img.shape[1])
                kp_y = int(detections[i, 4 + k*2 + 1] * img.shape[0])

                img = cv2.circle(img, (kp_x, kp_y), int(1.5*thickness), dot_color, -1)

    cv2.imwrite(save_image_path, img)

def decode_boxes(raw_boxes, back, anchors):
    """
    Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = np.zeros_like(raw_boxes)

    if back == True:
        x_scale = 256.0
        y_scale = 256.0
        h_scale = 256.0
        w_scale = 256.0
    else:
        x_scale = 128.0
        y_scale = 128.0
        h_scale = 128.0
        w_scale = 128.0

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(6):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] +\
            anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] +\
            anchors[:, 1]
        boxes[..., offset    ] = keypoint_x
        boxes[..., offset + 1] = keypoint_y
        
    return boxes


def intersect(box_a, box_b):
    """ We resize both arrays to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (array) bounding boxes, Shape: [A,4].
      box_b: (array) bounding boxes, Shape: [B,4].
    Return:
      (array) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    # max_xy = (np.expand_dims(box_a[:, 2:], axis=1).expand(A, B, 2),
    #           np.expand_dims(box_b[:, 2:], axis=0).expand(A, B, 2)).min()
    # min_xy = (np.expand_dims(box_a[:, :2], axis=1).expand(A, B, 2),
    #           np.expand_dims(box_b[:, :2], axis=0).expand(A, B, 2)).max()

    max_xy = np.minimum(
        np.repeat(np.expand_dims(box_a[:, 2:], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, 2:], axis=0), A, axis=0),
        
    )
    min_xy = np.maximum(
        np.repeat(np.expand_dims(box_a[:, :2], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, :2], axis=0), A, axis=0),
        
    )
    
    inter = np.clip((max_xy - min_xy), 0, None)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (array) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (array) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (array) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = np.repeat(
        np.expand_dims(
            (box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]),
            axis=1
        ),
        inter.shape[1],
        axis=1
    )  # [A,B]
    area_b = np.repeat(
        np.expand_dims(
            (box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1]),
            axis=0
        ),
        inter.shape[0],
        axis=0
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)


def weighted_non_max_suppression(detections):
    min_suppression_threshold = 0.3
    if len(detections) == 0:
        return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    # argsort(-x) returns the descending order version of argsort(x)
    remaining = np.argsort(-detections[:, 16])

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other 
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.copy()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :16]
            scores = detections[overlapping, 16:17]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(axis=0) / total_score
            weighted_detection[:16] = weighted
            weighted_detection[16] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections    


def postprocess(preds_ailia, anchor_path='anchors.npy', back=False, min_score_thresh = DEFAULT_MIN_SCORE_THRESH):
    raw_box = preds_ailia[0]  # (1, 896, 16)
    raw_score = preds_ailia[1]  # (1, 896, 1)

    anchors = np.load(anchor_path).astype(np.float32)
    score_thresh = 100.0
    
    detection_boxes = decode_boxes(raw_box, back, anchors)  # (1, 896, 16)
    raw_score = np.clip(raw_score, -score_thresh, score_thresh)  # (1, 896, 1)
    detection_scores = np.squeeze(sigmoid(raw_score), axis=-1)  # (1, 896)
    
    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= min_score_thresh  # (1, 896)

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    detections = []
    for i in range(raw_box.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = np.expand_dims(detection_scores[i, mask[i]], axis=-1)
        detections.append(np.concatenate((boxes, scores), axis=-1))

    # Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, 17))
        filtered_detections.append(faces)
    return filtered_detections


def compute_blazeface_with_keypoint(detector, frame, anchor_path='anchors.npy', back=False, min_score_thresh = DEFAULT_MIN_SCORE_THRESH):
    if back:
        BLAZEFACE_INPUT_IMAGE_HEIGHT = 256
        BLAZEFACE_INPUT_IMAGE_WIDTH = 256
    else:
        BLAZEFACE_INPUT_IMAGE_HEIGHT = 128
        BLAZEFACE_INPUT_IMAGE_WIDTH = 128

    # preprocessing
    image = letterbox_convert(frame, (BLAZEFACE_INPUT_IMAGE_HEIGHT, BLAZEFACE_INPUT_IMAGE_WIDTH))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))  # channel first
    image = image[np.newaxis, :, :, :]  # (batch_size, channel, h, w)
    input_data = image / 127.5 - 1.0

    # inference
    preds_ailia = detector.predict([input_data])

    # postprocessing
    face_detections = postprocess(preds_ailia, anchor_path, back=back, min_score_thresh=min_score_thresh)
    face_detections = face_detections[0]

    detections = []
    detections_eyes = []
    for i, d in enumerate(face_detections):
        # face position
        obj = ailia.DetectorObject(
            category=0,
            prob=1.0,
            x=d[1],
            y=d[0],
            w=d[3] - d[1],
            h=d[2] - d[0])
        detections.append(obj)

        # keypoints
        obj = ailia.DetectorObject(
            category=0,
            prob=1.0,
            x=d[4],
            y=d[5],
            w=0,
            h=0)
        detections_eyes.append(obj)

        obj = ailia.DetectorObject(
            category=0,
            prob=1.0,
            x=d[6],
            y=d[7],
            w=0,
            h=0)
        detections_eyes.append(obj)

    # revert square from detections
    detections = reverse_letterbox(detections, frame, (BLAZEFACE_INPUT_IMAGE_HEIGHT,BLAZEFACE_INPUT_IMAGE_WIDTH))
    detections_eyes = reverse_letterbox(detections_eyes, frame, (BLAZEFACE_INPUT_IMAGE_HEIGHT,BLAZEFACE_INPUT_IMAGE_WIDTH))

    # convert to keypoints
    keypoints = []
    for i in range(len(detections_eyes)//2):
        keypoint = {
            "eye_left_x": detections_eyes[i*2+0].x, "eye_left_y": detections_eyes[i*2+0].y,
            "eye_right_x": detections_eyes[i*2+1].x, "eye_right_y": detections_eyes[i*2+1].y
        }
        keypoints.append(keypoint)

    return detections, keypoints


def compute_blazeface(detector, frame, anchor_path='anchors.npy', back=False, min_score_thresh = DEFAULT_MIN_SCORE_THRESH):
    detections, keypoints = compute_blazeface_with_keypoint(detector, frame, anchor_path, back=back, min_score_thresh=min_score_thresh)
    return detections


def crop_blazeface(obj, margin, frame):
    w = frame.shape[1]
    h = frame.shape[0]
    cx = (obj.x + obj.w/2) * w
    cy = (obj.y + obj.h/2) * h
    cw = max(obj.w * w * margin, obj.h * h * margin)
    fx = max(cx - cw/2, 0)
    fy = max(cy - cw/2, 0)
    fw = min(cw, w-fx)
    fh = min(cw, h-fy)
    top_left = (int(fx), int(fy))
    bottom_right = (int((fx+fw)), int(fy+fh))
    crop_img = frame[
        top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 0:3
    ]
    return crop_img, top_left, bottom_right


def show_result(input_img, detections):
    for detection in detections:
        for d in detection:
            w = input_img.shape[1]
            h = input_img.shape[0]
            top_left = (int(d[1]*w), int(d[0]*h))
            bottom_right = (int(d[3]*w), int(d[2]*h))
            color = (255, 255, 255)
            cv2.rectangle(input_img, top_left, bottom_right, color, 4)

            for k in range(6):
                kp_x = d[4 + k*2] * input_img.shape[1]
                kp_y = d[4 + k*2 + 1] * input_img.shape[0]
                r = int(input_img.shape[1]/100)
                cv2.circle(input_img, (int(kp_x), int(kp_y)),
                           r, (255, 255, 255), -1)
