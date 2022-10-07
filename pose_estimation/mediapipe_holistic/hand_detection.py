import math

import cv2
import numpy as np

from image_utils import normalize_image
from math_utils import sigmoid

from detection_utils import get_anchor, decode_boxes, weighted_nms

HAND_CROP_SIZE = 256
HAND_LMK_SIZE = 224

onnx = False


def hand_estimate(img, hand_landmarks, models):
    im_h, im_w = img.shape[:2]

    threshold = 0.1
    accept = hand_landmarks[0, 3] > threshold

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
    center_x = x_middle
    center_y = y_middle

    box_size = np.sqrt(
        (x_middle - x_wrist) * (x_middle - x_wrist)
        + (y_middle - y_wrist) * (y_middle - y_wrist)) * 2.0

    angle = np.pi * 0.5 - math.atan2(-(y_middle - y_wrist), x_middle - x_wrist)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

    center_x = center_x / im_w
    center_y = center_y / im_h
    width = box_size / im_w
    height = box_size / im_h

    # print("1---", center_x, center_y, width, height)

    shift_x = 0
    shift_y = -0.1
    x_shift = (im_w * width * shift_x * math.cos(rotation) - im_h * height * shift_y * math.sin(rotation)) / im_w
    y_shift = (im_w * width * shift_x * math.sin(rotation) + im_h * height * shift_y * math.cos(rotation)) / im_h
    center_x += x_shift
    center_y += y_shift

    long_side = max(width * im_w, height * im_h)
    width = long_side / im_w * 2.7
    height = long_side / im_h * 2.7

    # print("2---", center_x, center_y, width, height)

    center = (center_x * im_w, center_y * im_h)
    rotated_rect = (center, (width * im_w, height * im_h), rotation * 180. / np.pi)
    pts1 = cv2.boxPoints(rotated_rect)

    h = w = HAND_CROP_SIZE
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # cv2.imwrite("hand.png", transformed)
    transformed = cv2.imread("hand_left_0.png")

    transformed = normalize_image(transformed, '255')
    transformed = transformed.transpose(2, 0, 1)  # HWC -> CHW
    transformed = np.expand_dims(transformed, axis=0)
    input = transformed.astype(np.float32)

    # -- hand_recrop_by_roi

    # Decodes the landmark tensors into a vector of landmarks

    # Adjusts landmarks (already normalized to [0.f, 1.f]) on
    # the letterboxed hand image

    # Projects the landmarks from the cropped hand image to the corresponding
    # locations on the full image before cropping

    # Converts hand landmarks to a detection that tightly encloses all landmarks
    # - LandmarksToDetectionCalculator

    # Converts hand detection into a rectangle based on center and scale alignment
    # points
    # - AlignmentPointsRectsCalculator

    # Slighly moves hand re-crop rectangle from wrist towards fingertips. Due to the
    # new hand cropping logic, crop border is to close to finger tips while a lot of
    # space is below the wrist. And when moving hand up fast (with fingers pointing
    # up) and using hand rect from the previous frame for tracking - fingertips can
    # be cropped. This adjustment partially solves it, but hand cropping logic
    # should be reviewed.
    # - RectTransformationCalculator

    # HandLandmarkCpu

    # Transforms a region of image into a 224x224 tensor while keeping the aspect
    # ratio, and therefore may result in potential letterboxing.

    # Converts the hand-flag tensor into a float that represents the confidence
    # score of hand presence.

    # Applies a threshold to the confidence score to determine whether a hand is
    # present.

    # Converts the handedness tensor into a float that represents the classification
    # score of handedness.

    # Decodes the landmark tensors into a list of landmarks, where the landmark
    # coordinates are normalized by the size of the input image to the model.
    # - TensorsToLandmarksCalculator

    # Adjusts landmarks (already normalized to [0.f, 1.f]) on the letterboxed hand
    # image (after image transformation with the FIT scale mode)
    # - LandmarkLetterboxRemovalCalculator

    # Projects the landmarks from the cropped hand image to the corresponding
    # locations on the full image before cropping (input to the graph).
    # - LandmarkProjectionCalculator

    x = np.array([])
    return x


def hands_estimate(img, left_hand_landmarks, right_hand_landmarks, models):
    left_hand_landmarks = hand_estimate(img, left_hand_landmarks, models)
    # right_hand_landmarks = hand_estimate(img, right_hand_landmarks, models)

    return left_hand_landmarks, right_hand_landmarks
