import cv2
import numpy as np
from scipy.special import expit


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

num_coords = 18
x_scale = 256.0
y_scale = 256.0
h_scale = 256.0
w_scale = 256.0
min_score_thresh = 0.75
min_suppression_threshold = 0.3
num_keypoints = 7

# mediapipe/graphs/hand_tracking/subgraphs/hand_detection_cpu.pbtxt
kp1 = 0
kp2 = 2
theta0 = np.pi/2
dscale = 2.6
dy = -0.5

resolution = 256


def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0] >= size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0]
        padh = 0
        padw = 256 - w1
        scale = size0[1] / w1
    else:
        h1 = 256 * size0[0] // size0[1]
        w1 = 256
        padh = 256 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh % 2
    padw1 = padw//2
    padw2 = padw//2 + padw % 2
    img1 = cv2.resize(img, (w1, h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0, 0)), mode='constant')
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128, 128))
    return img1, img2, scale, pad


def decode_boxes(raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(num_keypoints):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def raw_output_to_detections(raw_box, raw_score, anchors):
    """The output of the neural network is an array of shape (b, 896, 18)
    containing the bounding box regressor predictions, as well as an array
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" arrays into proper detections.
    Returns a list of (num_detections, 13) arrays, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    detection_boxes = decode_boxes(raw_box, anchors)

    thresh = 100.0
    raw_score = raw_score.clip(-thresh, thresh)
    # instead of defining our own sigmoid function which yields a warning
    # expit = sigmoid
    detection_scores = expit(raw_score).squeeze(axis=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= min_score_thresh

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = np.expand_dims(detection_scores[i, mask[i]], axis=-1)
        output_detections.append(np.concatenate((boxes, scores), axis=-1))

    return output_detections


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
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
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
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
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.

    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0:
        return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    # argsort() returns ascending order, therefore read the array from end
    remaining = np.argsort(detections[:, num_coords])[::-1]

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
            coordinates = detections[overlapping, :num_coords]
            scores = detections[overlapping, num_coords:num_coords+1]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(axis=0) / total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections


def denormalize_detections(detections, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        scale: scalar that was used to resize the image
        pad: padding in the x and y dimensions

    """
    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections


def detector_postprocess(preds_ailia, anchor_path='anchors.npy'):
    """
    Process detection predictions from ailia and return filtered detections
    """
    raw_box = preds_ailia[0]  # (1, 896, 18)
    raw_score = preds_ailia[1]  # (1, 896, 1)

    anchors = np.load(anchor_path).astype("float32")

    # Postprocess the raw predictions:
    detections = raw_output_to_detections(raw_box, raw_score, anchors)

    # Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, num_coords+1))
        filtered_detections.append(faces)

    return filtered_detections


def detection2roi(detection):
    """ Convert detections from detector to an oriented bounding box.

    Adapted from:
    # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

    The center and size of the box is calculated from the center
    of the detected box. Rotation is calcualted from the vector
    between kp1 and kp2 relative to theta0. The box is scaled
    and shifted by dscale and dy.

    """
    # compute box center and scale
    # use mediapipe/calculators/util/detections_to_rects_calculator.cc
    xc = (detection[:, 1] + detection[:, 3]) / 2
    yc = (detection[:, 0] + detection[:, 2]) / 2
    scale = (detection[:, 3] - detection[:, 1])  # assumes square boxes

    # compute box rotation
    x0 = detection[:, 4+2*kp1]
    y0 = detection[:, 4+2*kp1+1]
    x1 = detection[:, 4+2*kp2]
    y1 = detection[:, 4+2*kp2+1]
    theta = np.arctan2(y0-y1, x0-x1) - theta0

    center = np.stack((xc, yc), axis=1)
    dy_axis = np.column_stack((-np.sin(theta), np.cos(theta)))
    center += dy * scale[..., np.newaxis] * dy_axis
    xc, yc = center.T
    scale *= dscale

    return xc, yc, scale, theta


def extract_roi(frame, xc, yc, theta, scale):
    # take points on unit square and transform them according to the roi
    points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]]).reshape(1, 2, 4)
    points = points * scale.reshape(-1, 1, 1)/2
    theta = theta.reshape(-1, 1, 1)
    R = np.concatenate((
        np.concatenate((np.cos(theta), -np.sin(theta)), 2),
        np.concatenate((np.sin(theta), np.cos(theta)), 2),
    ), 1)
    center = np.concatenate((xc.reshape(-1, 1, 1), yc.reshape(-1, 1, 1)), 1)
    points = R @ points + center

    # use the points to compute the affine transform that maps
    # these points back to the output square
    res = resolution
    points1 = np.array([[0, 0, res-1], [0, res-1, 0]], dtype='float32').T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].T.astype('float32')
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res, res))  # , borderValue=127.5)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype('float32')
        affines.append(affine)
    if imgs:
        imgs = np.moveaxis(np.stack(imgs), 3, 1).astype('float32') / 255.
        affines = np.stack(affines)
    else:
        imgs = np.zeros((0, 3, res, res))
        affines = np.zeros((0, 2, 3))

    return imgs, affines, points


def estimator_preprocess(src_img, detections, scale, pad):
    """
    Extract ROI given detections
    """
    pose_detections = denormalize_detections(detections, scale, pad)
    xc, yc, scale, theta = detection2roi(pose_detections)
    img, affine, box = extract_roi(src_img, xc, yc, theta, scale)

    return img, affine, box


def denormalize_landmarks(normalized_landmarks, affines):
    landmarks = normalized_landmarks.copy()
    landmarks[:,:,:2] *= resolution
    for i in range(len(landmarks)):
        landmark, affine = landmarks[i], affines[i]
        landmark = (affine[:, :2] @ landmark[:, :2].T + affine[:, 2:]).T
        landmarks[i, :, :2] = landmark
    return landmarks

def normalize_radians(angle):
  return angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

def compute_rotation(landmarks):
    kWristJoint = 0
    kMiddleFingerPIPJoint = 6
    kIndexFingerPIPJoint = 4
    kRingFingerPIPJoint = 8
    kTargetAngle = np.pi * 0.5

    x0 = landmarks[kWristJoint, 0] * resolution
    y0 = landmarks[kWristJoint, 1] * resolution

    x1 = (landmarks[kIndexFingerPIPJoint, 0] + landmarks[kRingFingerPIPJoint, 0]) / 2
    y1 = (landmarks[kIndexFingerPIPJoint, 1] + landmarks[kRingFingerPIPJoint, 1]) / 2
    x1 = (x1 + landmarks[kMiddleFingerPIPJoint, 0]) / 2 * resolution
    y1 = (y1 + landmarks[kMiddleFingerPIPJoint, 1]) / 2 * resolution

    rotation = normalize_radians(kTargetAngle - np.arctan2(-(y1 - y0), x1 - x0))
    return rotation

def landmarks2roi(landmarks, affine):
    """
    Inputs:
        landmarks: normalized cropped hand image landmarks
        affine: Affine transform matrix to get original coordinates from
            cropped image coordinates
    
    Outputs:
        ROI x center, y center, scale (width = height), theta (rotation)
        in original image coordinates

    Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt
    """
    partial_landmarks_id = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    partial_landmarks = landmarks[partial_landmarks_id, :]

    rotation = compute_rotation(partial_landmarks)
    c, s = np.cos(rotation), np.sin(rotation)
    rot_mat = np.array([
        [ c, -s],
        [ s,  c]
    ])
    rev_rot_mat = np.array([
        [ c,  s],
        [-s,  c]
    ])

    # Find boundaries of landmarks.
    axis_min = np.min(partial_landmarks[:, :2], axis=0)
    axis_max = np.max(partial_landmarks[:, :2], axis=0)
    axis_aligned_center = (axis_min + axis_max) / 2

    # Find boundaries of rotated landmarks.
    translated = (partial_landmarks[:, :2] - axis_aligned_center[None]) * resolution
    projected = translated @ rev_rot_mat.T
    projected_min = np.min(projected, axis=0)
    projected_max = np.max(projected, axis=0)

    scale = [2, 2]
    shift = [0, -0.1]

    projected_center = (projected_min + projected_max) / 2
    projected_wh = projected_max - projected_min
    projected_center += projected_wh * shift

    long_side = np.max(projected_wh)
    projected_wh[:] = long_side
    projected_wh *= scale

    projected_min_x, projected_min_y = projected_center - projected_wh / 2
    projected_max_x, projected_max_y = projected_center + projected_wh / 2

    projected_corners = np.array([
        [projected_min_x, projected_min_y], # top left
        [projected_max_x, projected_min_y], # top right
        [projected_min_x, projected_max_y], # bottom left
        [projected_max_x, projected_max_y]  # bottom right
    ])

    # Corners in cropped hand image coordinates
    cropped_corners = (projected_corners @ rot_mat.T) + axis_aligned_center[None] * resolution

    # Compute ROI in original image coordinates
    corners = (affine[:,:2] @ cropped_corners.T + affine[:,2:]).T
    x_center, y_center = corners.mean(axis=0)
    scale_orig = np.linalg.norm(corners[1] - corners[0])
    theta = np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0])

    return x_center, y_center, scale_orig, theta
