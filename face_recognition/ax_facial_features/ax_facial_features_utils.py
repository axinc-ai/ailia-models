import cv2
import numpy as np
from scipy.special import expit

import sys
sys.path.append('../../util')

num_coords = 16
x_scale = 128.0
y_scale = 128.0
h_scale = 128.0
w_scale = 128.0
min_score_thresh = 0.75
min_suppression_threshold = 0.3
num_keypoints = 6

# mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
kp1 = 1  # Left eye
kp2 = 0  # Right eye
theta0 = 0
dscale = 1.5
dy = 0.

resolution = 192

EYE_LEFT_CONTOUR = [
    249, 263, 362, 373, 374,
    380, 381, 382, 384, 385,
    386, 387, 388, 390, 398, 466
]
EYE_RIGHT_CONTOUR = [
    7, 33, 133, 144, 145,
    153, 154, 155, 157, 158,
    159, 160, 161, 163, 173, 246
]
MOUTH_INNER_CONTOUR = [
    78, 191, 80, 81, 82,
    13, 312, 311, 310, 415,
    308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,
]


def resize_image(img, out_size, keep_aspect_ratio=True, return_scale_padding=False):
    """
    Resizes the input image to the desired size, keeping the original aspect
    ratio or not.

    Parameters
    ----------
    img: NumPy array
        The image to resize.
    out_size: int or (int, int)  (height, width)
        Resizes the image to the desired size.
    keep_aspect_ratio: bool (default: True)
        If true, resizes while keeping the original aspect ratio. Adds zero-
        padding if necessary.
    return_scale_padding: bool (default: False)
        If true, returns the scale and padding for each dimensions.

    Returns
    -------
    resized: NumPy array
        Resized image.
    scale: NumPy array, optional
        Resized / original, (scale_height, scale_width).
    padding: NumPy array, optional
        Zero padding (top, bottom, left, right) added after resizing.
    """
    img_size = img.shape[:2]
    if isinstance(out_size, int):
        out_size = np.array([out_size, out_size], dtype=int)
    else: # Assuming sequence of len 2
        out_size = np.array(out_size, dtype=int)
    scale = img_size / out_size
    padding = np.zeros(4, dtype=int)

    if img_size[0] != img_size[1] and keep_aspect_ratio:
        scale_long_side = np.max(scale)
        size_new = (img_size / scale_long_side).astype(int)
        padding = out_size - size_new
        padding = np.stack((padding // 2, padding - padding // 2), axis=1).flatten()
        scale[:] = scale_long_side
        resized = cv2.resize(img, (size_new[1], size_new[0]))
        resized = cv2.copyMakeBorder(resized, *padding, cv2.BORDER_CONSTANT, 0)
    else:
        resized = cv2.resize(img, (out_size[1], out_size[0]))

    if return_scale_padding:
        return resized, scale, padding
    else:
        return resized


def face_detector_preprocess(img):
    """Preprocesses the image for the face detector.

    Parameters
    ----------
    img: NumPy array
        The image to format in BGR channel order.

    Returns
    -------
    input_face_det: NumPy array
        Formatted image.
    scale: NumPy array
        Resized / original, (scale_height, scale_width)
    padding: NumPy array
        Zero padding (top, bottom, left, right) added after resizing
    """
    input_face_det, scale, padding = resize_image(img[..., ::-1], 128, return_scale_padding=True)
    input_face_det = input_face_det.astype(np.float32) / 127.5 - 1.0
    input_face_det = np.moveaxis(input_face_det, -1, 0)[np.newaxis]
    return input_face_det, scale, padding


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
    """The output of the neural network is an array of shape (b, 896, 16)
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
    # (instead of defining our own sigmoid function which yields a warning)
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
            axis=1,
        ),
        inter.shape[1],
        axis=1,
    )  # [A,B]
    area_b = np.repeat(
        np.expand_dims(
            (box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1]),
            axis=0,
        ),
        inter.shape[0],
        axis=0,
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


def face_detector_postprocess(preds, anchor_path='anchors.npy'):
    """Process detection predictions and return filtered detections"""
    raw_box = preds[0]  # (1, 896, 16)
    raw_score = preds[1]  # (1, 896, 1)

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


def denormalize_detections(detections, resized_size, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        resized_size: size of the resized image (i.e. input image)
        scale: scalar that was used to resize the image
        pad: padding in the x (left) and y (top) dimensions

    """
    detections[:, 0] = (detections[:, 0] * resized_size - pad[0]) * scale
    detections[:, 1] = (detections[:, 1] * resized_size - pad[1]) * scale
    detections[:, 2] = (detections[:, 2] * resized_size - pad[0]) * scale
    detections[:, 3] = (detections[:, 3] * resized_size - pad[1]) * scale

    detections[:, 4::2] = (detections[:, 4::2] * resized_size - pad[1]) * scale
    detections[:, 5::2] = (detections[:, 5::2] * resized_size - pad[0]) * scale
    return detections


def detection2roi(detection, detection2roi_method='box'):
    """ Convert detections from detector to an oriented bounding box.

    Adapted from:
    mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

    The center and size of the box is calculated from the center
    of the detected box. Rotation is calculated from the vector
    between kp1 and kp2 relative to theta0. The box is scaled
    and shifted by dscale and dy.

    """
    if detection2roi_method == 'box':
        # compute box center and scale
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        xc = (detection[:, 1] + detection[:, 3]) / 2
        yc = (detection[:, 0] + detection[:, 2]) / 2
        scale = (detection[:, 3] - detection[:, 1])  # assumes square boxes

    elif detection2roi_method == 'alignment':
        # compute box center and scale
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        xc = detection[:, 4+2*kp1]
        yc = detection[:, 4+2*kp1+1]
        x1 = detection[:, 4+2*kp2]
        y1 = detection[:, 4+2*kp2+1]
        scale = np.sqrt(((xc-x1)**2 + (yc-y1)**2)) * 2
    else:
        raise NotImplementedError(
            "detection2roi_method [%s] not supported" % detection2roi_method)

    yc += dy * scale
    scale *= dscale

    # compute box rotation
    x0 = detection[:, 4+2*kp1]
    y0 = detection[:, 4+2*kp1+1]
    x1 = detection[:, 4+2*kp2]
    y1 = detection[:, 4+2*kp2+1]
    theta = np.arctan2(y0-y1, x0-x1) - theta0
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
        img = cv2.warpAffine(frame, M, (res, res), borderValue=127.5)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype('float32')
        affines.append(affine)
    if imgs:
        imgs = np.moveaxis(np.stack(imgs), 3, 1).astype('float32') / 127.5 - 1.0
        affines = np.stack(affines)
    else:
        imgs = np.zeros((0, 3, res, res))
        affines = np.zeros((0, 2, 3))

    return imgs, affines, points


def face_lm_preprocess(img, detections, scale, padding):
    """Preprocesses the image and face detections for the face landmarks estimator.

    Parameters
    ----------
    img: NumPy array
        The image to format in BGR channel order.
    detections: NumPy array
        Face detections.
    scale: NumPy array
        Scale used when preprocessing the image for the face detection.
        Resized / original, (scale_height, scale_width)
    padding: NumPy array
        Padding used when preprocessing the image for the face detection.
        Zero padding (top, bottom, left, right) added after resizing

    Returns
    -------
    input_face_lm: NumPy array
        Formatted image.
    affines: NumPy array
        Affine transform that maps points in the cropped 192x192 image back to
        the original image
    centers: NumPy array
        Center(s) (x, y) of the cropped faces.
    theta: NumPy array
        rotation angle(s) in radians of the cropping bounding boxes.
    """
    # Only handles detections from the 1st image
    detections = denormalize_detections(detections[0], 128, scale[0], padding[[0, 2]])
    xc, yc, roi_scale, theta = detection2roi(detections)
    input_face_lm, affine, _ = extract_roi(img[..., ::-1], xc, yc, theta, roi_scale)
    centers = np.stack((xc, yc), axis=1)

    return input_face_lm, affine, centers, theta


def denormalize_landmarks(landmarks, affines):
    landmarks = landmarks.reshape((landmarks.shape[0], -1, 3))
    landmarks[:, :, :] *= resolution
    for i in range(len(landmarks)):
        landmark, affine = landmarks[i], affines[i]
        landmark = (affine[:, :2] @ landmark[:, :2].T + affine[:, 2:]).T
        landmarks[i, :, :2] = landmark
    return landmarks


def face_lm_postprocess(preds, affines):
    """Filter face landmarks given confidence and denormalize.

    Parameters
    ----------
    preds: tuple of NumPy array
        Facemesh predictions (raw face landmark predictions, raw confidence
        values).
    affines: NumPy array
        Affine transform that maps points in the cropped 192x192 image back to
        the original image

    Returns
    -------
    landmarks_: NumPy array
        Filtered landmarks.
    confidences_: NumPy array
        Filtered confidences.
    affines_: NumPy array
        Filtered affine transforms.
    """
    landmarks_ = preds[0].reshape((-1, 1404))
    # Raw confidence can be converted to score by applying sigmoid
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt
    confidences_ = expit(preds[1].reshape((-1, 1)))
    mask = confidences_.squeeze(axis=1) > 0.5
    landmarks_ = landmarks_[mask].copy()
    confidences_ = confidences_[mask]
    affines_ = affines[mask]
    if len(landmarks_) > 0:
        landmarks_ = denormalize_landmarks(landmarks_ / resolution, affines_)

    return landmarks_, confidences_, affines_


def compute_centers(landmarks):
    """Compute left & right eye centers and mouth center"""
    b = landmarks.shape[0]
    lms = landmarks.reshape((b, -1, 3))

    eye_left_centers = lms[:, EYE_LEFT_CONTOUR, :2].mean(axis=1)
    eye_right_centers = lms[:, EYE_RIGHT_CONTOUR, :2].mean(axis=1)
    mouth_centers = lms[:, MOUTH_INNER_CONTOUR, :2].mean(axis=1)

    a = np.concatenate((eye_left_centers, eye_right_centers, mouth_centers), axis=1)

    return a


def compute_corners(landmarks, mode):
    """Compute the four corners of the crop

    Top left, top right, bottom left, bottom right    
    """
    right_to_left_eye = landmarks[:, :2] - landmarks[:, 2:4]
    middle_eye = (landmarks[:, :2] + landmarks[:, 2:4]) / 2
    eye_to_mouth = landmarks[:, 4:6] - middle_eye
    centers = middle_eye

    if np.linalg.norm(right_to_left_eye) > np.linalg.norm(eye_to_mouth):
        vec_right = right_to_left_eye
        vec_down = np.fliplr(vec_right).copy()
        vec_down[:, 0] *= -1.
    else:
        vec_down = eye_to_mouth
        vec_right = np.fliplr(vec_down).copy()
        vec_right[:, 1] *= -1.

    if mode == 'face':
        scale = 1.8
    elif mode == 'eyes':
        vec_down *= 0.33
        scale = 1.
    else:
        raise NotImplementedError()

    diag = scale * (vec_right + vec_down)
    top_left = centers - diag
    top_right = top_left + 2 * scale * vec_right
    bottom_left = top_left + 2 * scale * vec_down
    bottom_right = centers + diag

    return top_left, top_right, bottom_left, bottom_right


def get_roi(landmarks):
    """Get the corners of the ROI
    
    Top left, top right, bottom left, bottom right 
    """
    centers = compute_centers(landmarks)
    corners = compute_corners(centers, 'face')
    return [corners[i][0] for i in range(len(corners))]


def crop(img, centers, mode):
    """Get a face or eye region crop
    
    Only return the first crop
    """
    h, w, c = img.shape
    corners = compute_corners(centers, mode)
    top_left, top_right, bottom_left, bottom_right = corners

    # New shape for rotated crop
    f_px_norm = lambda x: np.round(np.linalg.norm(x, axis=1)).astype(np.int32)
    new_h = f_px_norm(bottom_left - top_left)
    new_w = f_px_norm(top_right - top_left)

    crops = []
    for i in range(len(centers)):
        tl = top_left[i]
        tr = top_right[i]
        bl = bottom_left[i]
        br = bottom_right[i]

        # Get a rotated crop centered on each face
        corners = np.stack((tl, tr, bl)).astype(np.float32)
        new_corners = np.asarray([[0., 0.], [new_w[i], 0.], [0., new_h[i]]],
                                 dtype=np.float32)
        M = cv2.getAffineTransform(corners, new_corners)
        crop = cv2.warpAffine(img, M, (new_w[i], new_h[i]), flags=cv2.INTER_LANCZOS4)
        crops.append(crop)

    return crops[0]


def facial_features_preprocess(img, landmarks, mode):
    """Preprocess the image for the facial feature model"""
    centers = compute_centers(landmarks)
    crop_img = crop(img, centers, mode)
    crop_img = crop_img[..., ::-1]
    in_data = resize_image(crop_img, 256, keep_aspect_ratio=False)
    in_data = in_data.astype(np.float32) / 255.
    in_data = np.moveaxis(in_data, -1, 0)[np.newaxis]
    m = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
    s = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))
    in_data = (in_data - m) / s
    return in_data


def draw_roi(img, roi):
    """Draw the ROI on the image"""
    (x1, x2, x3, x4), (y1, y2, y3, y4) = roi.T
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0, 0, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 0), 2)


def filter_sort_results(scores, labels, multilabel=False, max_class_count=3):
    """Filter and sort the results
    
    - Descending order for multiclass classification
    - Order of appearance for multi-label classification
    """
    if multilabel:
        assert len(scores) == len(labels)
        max_class_count = len(labels)
        ids_order = range(max_class_count)
    else:
        max_class_count = min(len(labels), max_class_count)
        ids_order = np.argsort(scores)[::-1][:max_class_count]
    return ids_order


def print_results(
    scores, labels, logger, multilabel=False, max_class_count=3
):
    """Print classification results"""
    ids_order = filter_sort_results(
        scores, labels, multilabel=multilabel, max_class_count=max_class_count
    )

    logger.info('==============================================================')
    if multilabel:
        logger.info(f'label_count = {len(ids_order)}')
    else:
        logger.info(f'class_count = {len(ids_order)}')
    for i, idx in enumerate(ids_order):
        tmp = f'category = {idx} [{labels[idx]}]'
        if multilabel:
            logger.info(f'+ {tmp}')
        else:
            logger.info(f'+ idx = {i}')
            logger.info(f'  {tmp}')
        logger.info(f'  prob = {scores[idx]}')
    logger.info('')
