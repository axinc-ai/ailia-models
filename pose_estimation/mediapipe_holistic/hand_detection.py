import math

import cv2
import numpy as np

from image_utils import normalize_image

HAND_CROP_SIZE = 256
HAND_LMK_SIZE = 224

onnx = False


def hand_estimate(img, hand_landmarks, models):
    im_h, im_w = img.shape[:2]

    # Drops hand-related pose landmarks if pose wrist is not visible. It will
    # prevent from predicting hand landmarks on the current frame.
    threshold = 0.1
    accept = hand_landmarks[0, 3] > threshold
    if not accept:
        return np.zeros((0, 3)), np.zeros((0, 2))

    # Converts hand-related pose landmarks to a detection that tightly encloses all
    # of them.
    x_wrist = hand_landmarks[0, 0] * im_w
    y_wrist = hand_landmarks[0, 1] * im_h
    x_index = hand_landmarks[2, 0] * im_w
    y_index = hand_landmarks[2, 1] * im_h
    x_pinky = hand_landmarks[1, 0] * im_w
    y_pinky = hand_landmarks[1, 1] * im_h

    # Estimate middle finger
    x_middle = (2. * x_index + x_pinky) / 3.
    y_middle = (2. * y_index + y_pinky) / 3.

    # Crop center as middle finger
    x_center = x_middle
    y_center = y_middle

    # Converts hand detection to a normalized hand rectangle.
    box_size = np.sqrt(
        (x_middle - x_wrist) * (x_middle - x_wrist)
        + (y_middle - y_wrist) * (y_middle - y_wrist)) * 2.0

    angle = np.pi * 0.5 - math.atan2(-(y_middle - y_wrist), x_middle - x_wrist)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

    x_center = x_center / im_w
    y_center = y_center / im_h
    width = box_size / im_w
    height = box_size / im_h

    # Expands the palm rectangle so that it becomes big enough for hand re-crop
    # model to localize it accurately.
    shift_x = 0
    shift_y = -0.1
    x_shift = (im_w * width * shift_x * math.cos(rotation) - im_h * height * shift_y * math.sin(rotation)) / im_w
    y_shift = (im_w * width * shift_x * math.sin(rotation) + im_h * height * shift_y * math.cos(rotation)) / im_h
    x_center += x_shift
    y_center += y_shift

    long_side = max(width * im_w, height * im_h)
    width = long_side / im_w * 2.7
    height = long_side / im_h * 2.7

    center = (x_center * im_w, y_center * im_h)
    rotated_rect = (center, (width * im_w, height * im_h), rotation * 180. / np.pi)
    pts1 = cv2.boxPoints(rotated_rect)

    h = w = HAND_CROP_SIZE
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    transformed = normalize_image(transformed, '255')
    transformed = transformed.transpose(2, 0, 1)  # HWC -> CHW
    transformed = np.expand_dims(transformed, axis=0)
    input = transformed.astype(np.float32)

    # Predicts hand re-crop rectangle
    net = models['hand_det']
    if not onnx:
        output = net.predict([input])
    else:
        output = net.run(None, {'input_1': input})
    output_crop = output[0]

    landmarks = output_crop.reshape(-1, 2) / HAND_CROP_SIZE

    # Projects the landmarks from the cropped hand image to the corresponding
    # locations on the full image before cropping
    def project_fn(x, y):
        x -= 0.5
        y -= 0.5
        new_x = math.cos(rotation) * x - math.sin(rotation) * y
        new_y = math.sin(rotation) * x + math.cos(rotation) * y
        new_x = new_x * width + x_center
        new_y = new_y * height + y_center
        return new_x, new_y

    for lmks in landmarks:
        x, y = lmks
        lmks[...] = project_fn(x, y)

    # Converts hand detection into a rectangle based on center and scale alignment
    # points
    x0, x1 = landmarks[:, 0] * im_w
    y0, y1 = landmarks[:, 1] * im_h
    box_size = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    box_size *= 2
    angle = (np.pi * -90 / 180) - math.atan2(-(y1 - y0), x1 - x0)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))
    ## ROI
    x_center = x0 / im_w
    y_center = y0 / im_h
    width = box_size / im_w
    height = box_size / im_h

    # Slighly moves hand re-crop rectangle from wrist towards fingertips. Due to the
    # new hand cropping logic, crop border is to close to finger tips while a lot of
    # space is below the wrist. And when moving hand up fast (with fingers pointing
    # up) and using hand rect from the previous frame for tracking - fingertips can
    # be cropped.
    shift_x = 0
    shift_y = -0.1
    x_shift = (im_w * width * shift_x * math.cos(rotation) - im_h * height * shift_y * math.sin(rotation)) / im_w
    y_shift = (im_w * width * shift_x * math.sin(rotation) + im_h * height * shift_y * math.cos(rotation)) / im_h
    x_center += x_shift
    y_center += y_shift

    long_side = max(width * im_w, height * im_h)
    width = long_side / im_w
    height = long_side / im_h

    center = (x_center * im_w, y_center * im_h)
    rotated_rect = (center, (width * im_w, height * im_h), rotation * 180. / np.pi)
    pts1 = cv2.boxPoints(rotated_rect)

    h = w = HAND_LMK_SIZE
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    transformed = normalize_image(transformed, '255')
    transformed = transformed.transpose(2, 0, 1)  # HWC -> CHW
    transformed = np.expand_dims(transformed, axis=0)
    input = transformed.astype(np.float32)

    # feedforward
    net = models['hand_lmk']
    if not onnx:
        output = net.predict([input])
    else:
        output = net.run(None, {'input_1': input})
    landmarks, hand_flag, handedness, world_landmarks = output

    # Applies a threshold to the confidence score to determine whether a hand is
    # present.
    threshold = 0.5
    accept = hand_flag[0, 0] > threshold
    if not accept:
        return np.zeros((0, 3)), np.zeros((0, 2))

    # Decodes the landmark tensors into a list of landmarks, where the landmark
    # coordinates are normalized by the size of the input image to the model.
    landmarks = landmarks.reshape(-1, 3) / HAND_LMK_SIZE
    world_landmarks = world_landmarks.reshape(-1, 3)

    normalize_z = 0.4
    landmarks[:, 2] = landmarks[:, 2] / normalize_z

    # Projects the landmarks from the cropped hand image to the corresponding
    # locations on the full image before cropping (input to the graph).
    cosa = math.cos(rotation)
    sina = math.sin(rotation)

    def project_fn(x, y, z):
        x -= 0.5
        y -= 0.5
        new_x = cosa * x - sina * y
        new_y = sina * x + cosa * y
        new_x = new_x * width + x_center
        new_y = new_y * height + y_center
        new_z = z * width  # Scale Z coordinate as X.
        return new_x, new_y, new_z

    for lmks in landmarks:
        x, y, z = lmks
        lmks[...] = project_fn(x, y, z)

    def project_fn(x, y):
        new_x = cosa * x - sina * y
        new_y = sina * x + cosa * y
        return new_x, new_y

    for lmks in world_landmarks:
        x, y, _ = lmks
        lmks[:2] = project_fn(x, y)

    return landmarks, world_landmarks


def hands_estimate(img, left_hand_landmarks, right_hand_landmarks, models):
    left_hand_landmarks, _ = hand_estimate(img, left_hand_landmarks, models)
    right_hand_landmarks, _ = hand_estimate(img, right_hand_landmarks, models)

    return left_hand_landmarks, right_hand_landmarks
