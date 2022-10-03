import math

import cv2
import numpy as np

from image_utils import normalize_image
from math_utils import sigmoid

from detection_utils import get_anchor, decode_boxes

IMAGE_FACE_SIZE = 128

onnx = False

face_anchors = get_anchor(
    num_layers=4,
    strides=[8, 16, 16, 16],
    input_size_height=IMAGE_FACE_SIZE,
    input_size_width=IMAGE_FACE_SIZE)


def predict_face_mesh(img, face_landmarks, models):
    im_h, im_w = img.shape[:2]

    xmin = np.min(face_landmarks[:, 0])
    xmax = np.max(face_landmarks[:, 0])
    ymin = np.min(face_landmarks[:, 1])
    ymax = np.max(face_landmarks[:, 1])
    width = xmax - xmin
    height = ymax - ymin

    x_center, y_center = (xmin + width / 2), (ymin + height / 2)

    # Converts face detection to a normalized face rectangle.
    start_keypoint_index = 5  # Right eye
    end_keypoint_index = 2  # Left eye
    x0 = face_landmarks[start_keypoint_index, 0] * im_w
    y0 = face_landmarks[start_keypoint_index, 1] * im_h
    x1 = face_landmarks[end_keypoint_index, 0] * im_w
    y1 = face_landmarks[end_keypoint_index, 1] * im_h
    angle = -math.atan2(-(y1 - y0), x1 - x0)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

    # Expands face rectangle so that it becomes big enough for face detector to
    # localize it accurately.
    long_side = max(width * im_w, height * im_h)
    width = long_side / im_w * 3
    height = long_side / im_h * 3

    width, height = width * im_w, height * im_h
    center = (x_center * im_w, y_center * im_h)
    rotated_rect = (center, (width, height), rotation * 180. / np.pi)
    pts1 = cv2.boxPoints(rotated_rect)

    h = w = IMAGE_FACE_SIZE
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # print(transformed)
    # print(transformed.shape)
    # cv2.imwrite("face_mesh.png", transformed)
    transformed = cv2.imread("face_mesh_0.png")

    transformed = normalize_image(transformed, '127.5')
    transformed = transformed.transpose(2, 0, 1)  # HWC -> CHW
    transformed = np.expand_dims(transformed, axis=0)
    input = transformed.astype(np.float32)

    # feedforward
    net = models['face_net']
    if not onnx:
        output = net.predict([input])
    else:
        output = net.run(None, {'input': input})
    scores, detections = output

    num_boxes = 896
    num_coords = 16
    num_keypoints = 6
    scale = IMAGE_FACE_SIZE
    boxes = decode_boxes(detections[0], face_anchors, num_boxes, num_coords, num_keypoints, scale)

    scores = np.clip(scores[0, :, 0], -100, 100)
    scores = sigmoid(scores)

    min_score_thresh = 0.5
    idx = scores >= min_score_thresh
    print(np.nonzero(idx))
    boxes = boxes[idx]
    scores = scores[idx]

    print(boxes)
    print(boxes.shape)
