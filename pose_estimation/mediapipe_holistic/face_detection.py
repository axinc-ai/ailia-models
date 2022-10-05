import math

import cv2
import numpy as np

from image_utils import normalize_image
from math_utils import sigmoid

from detection_utils import get_anchor, decode_boxes, weighted_nms

IMAGE_FACE_SIZE = 128

onnx = False

anchors = get_anchor(
    num_layers=4,
    strides=[8, 16, 16, 16],
    input_height=IMAGE_FACE_SIZE,
    input_width=IMAGE_FACE_SIZE)


def predict_face_mesh(img, face_landmarks, models):
    im_h, im_w = img.shape[:2]

    xmin = np.min(face_landmarks[:, 0])
    xmax = np.max(face_landmarks[:, 0])
    ymin = np.min(face_landmarks[:, 1])
    ymax = np.max(face_landmarks[:, 1])

    # ROI
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

    # matrix
    a = rotated_rect[1][0]
    b = rotated_rect[1][1]
    c = math.cos(rotation)
    d = math.sin(rotation)
    flip = 1
    e = rotated_rect[0][0]
    f = rotated_rect[0][1]
    g = 1 / im_w
    h = 1 / im_h
    transform_matrix = np.array([
        [a * c * flip * g, -b * d * g, 0, (-0.5 * a * c * flip + 0.5 * b * d + e) * g],
        [a * d * flip * h, b * c * h, 0, (-0.5 * b * c - 0.5 * a * d * flip + f) * h],
        [0, 0, a * g, 0],
        [0, 0, 0, 1]
    ])

    def project_fn(x, y):
        return transform_matrix[:2, [0, 1, 3]] @ np.array([x, y, 1])

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
    img_size = IMAGE_FACE_SIZE
    boxes = decode_boxes(detections[0], anchors, num_boxes, num_coords, num_keypoints, img_size)

    scores = np.clip(scores[0, :, 0], -100, 100)
    scores = sigmoid(scores)

    min_score_thresh = 0.5
    idx = scores >= min_score_thresh
    # print(np.nonzero(idx))
    boxes = boxes[idx]
    scores = scores[idx]

    # Performs non-max suppression to remove excessive detections.
    boxes, scores = weighted_nms(boxes, scores, img_size)

    # DetectionProjection
    for i in range(len(boxes)):
        a = boxes[i]
        for j in range(0, len(a[4:]), 2):
            x, y = a[j + 4:j + 6]
            x, y = project_fn(x, y)
            a[j + 4:j + 6] = x, y

        xmin, ymin, xmax, ymax = a[:4]
        c = [
            [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
        ]
        c = np.stack([project_fn(x[0], x[1]) for x in c])
        xmin, ymin = np.min(c, axis=0)
        xmax, ymax = np.max(c, axis=0)
        a[:4] = xmin, ymin, xmax, ymax

    # NormRectFromKeyPoints
    keys_x = boxes[0, list(range(4, 4 + num_keypoints * 2, 2))]
    keys_y = boxes[0, list(range(5, 5 + num_keypoints * 2, 2))]
    xmin = np.min(keys_x)
    ymin = np.min(keys_y)
    xmax = np.max(keys_x)
    ymax = np.max(keys_y)

    # Right eye
    x0 = boxes[0, 4] * im_w
    y0 = boxes[0, 5] * im_h
    # Left eye
    x1 = boxes[0, 6] * im_w
    y1 = boxes[0, 7] * im_h
    angle = -math.atan2(-(y1 - y0), x1 - x0)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

    # ROI
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    # TransformNormRect
    shift_x = 0
    shift_y = -0.1
    x_shift = (im_w * width * shift_x * math.cos(rotation) - im_h * height * shift_y * math.sin(rotation)) / im_w
    y_shift = (im_w * width * shift_x * math.sin(rotation) + im_h * height * shift_y * math.cos(rotation)) / im_h
    x_center += x_shift
    y_center += y_shift

    # Expands face rectangle so that it becomes big enough for face detector to
    # localize it accurately.
    long_side = max(width * im_w, height * im_h)
    width = long_side / im_w * 2
    height = long_side / im_h * 2

    # width, height = width * im_w, height * im_h
    # center = (x_center * im_w, y_center * im_h)
    # rotated_rect = (center, (width, height), rotation * 180. / np.pi)
    # pts1 = cv2.boxPoints(rotated_rect)
    print("rect ---", x_center, y_center, width, height)

    # FaceLandmarkCpu
