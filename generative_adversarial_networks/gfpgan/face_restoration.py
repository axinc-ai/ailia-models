from itertools import product

import numpy as np
import cv2

from nms_utils import nms_boxes
from math import ceil


def get_anchor(image_size):
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0]/step), ceil(image_size[1]/step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        m_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in m_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors.extend([cx, cy, s_kx, s_ky])

    output = np.array(anchors).reshape(-1, 4)
    return output


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
         priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    tmp = (
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
    )
    landms = np.concatenate(tmp, axis=1)

    return landms


def detect_faces(
        net,
        image,
        conf_threshold=0.8,
        nms_threshold=0.4):
    image = image - np.array([104., 117., 123.])
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    height, width = image.shape[2:]

    # feedforward
    output = net.predict([image])
    loc, conf, landmarks = output

    priors = get_anchor((height, width))

    variance = [0.1, 0.2]
    scale = np.array([width, height, width, height])
    scale1 = np.array([
        width, height, width, height, width, height, width, height, width, height
    ])

    # bboxe
    boxes = decode(loc[0], priors, variance)
    boxes = boxes * scale
    # score
    scores = conf[0][:, 1]
    # landmark
    landmarks = decode_landm(landmarks[0], priors, variance)
    landmarks = landmarks * scale1

    # ignore low scores
    inds = np.where(scores > conf_threshold)[0]
    boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

    # sort
    order = scores.argsort()[::-1]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # do NMS
    bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms_boxes(bounding_boxes[:, :4], bounding_boxes[:, 4], nms_threshold)
    bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]

    return np.concatenate((bounding_boxes, landmarks), axis=1)


def get_face_landmarks_5(
        net,
        input_img,
        eye_dist_threshold=None):
    bboxes = detect_faces(net, input_img, 0.97)

    all_landmarks_5 = []
    det_faces = []
    for bbox in bboxes:
        # remove faces with too small eye distance: side faces or too small faces
        eye_dist = np.linalg.norm([bbox[6] - bbox[8], bbox[7] - bbox[9]])
        if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
            continue

        landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
        all_landmarks_5.append(landmark)
        det_faces.append(bbox[0:5])

    return det_faces, all_landmarks_5


def align_warp_face(
        input_img, all_landmarks_5, face_size=512):
    """Align and warp faces with face template.
    """

    face_template = np.array([
        [192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
        [201.26117, 371.41043], [313.08905, 371.15118]
    ])
    face_template = face_template * (face_size / 512.0)

    cropped_faces = []
    affine_matrices = []
    for idx, landmark in enumerate(all_landmarks_5):
        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
        affine_matrices.append(affine_matrix)
        # warp and crop faces
        border_mode = cv2.BORDER_CONSTANT
        cropped_face = cv2.warpAffine(
            input_img, affine_matrix, (face_size, face_size),
            borderMode=border_mode,
            borderValue=(135, 133, 132))  # gray
        cropped_faces.append(cropped_face)

    return cropped_faces, affine_matrices


def get_inverse_affine(affine_matrices, upscale_factor=1):
    """Get inverse affine matrix."""
    inverse_affine_matrices = []
    for idx, affine_matrix in enumerate(affine_matrices):
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= upscale_factor
        inverse_affine_matrices.append(inverse_affine)

    return inverse_affine_matrices


def paste_faces_to_image(
        img,
        restored_faces, inverse_affine_matrices,
        upscale_factor=1,
        face_size=512):
    h, w, _ = img.shape
    for restored_face, inverse_affine in zip(restored_faces, inverse_affine_matrices):
        # Add an offset to inverse affine matrix, for more precise back alignment
        if upscale_factor > 1:
            extra_offset = 0.5 * upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w, h))

        mask = np.ones((face_size, face_size), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
        # remove the black borders
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((int(2 * upscale_factor), int(2 * upscale_factor)), np.uint8))
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)  # // 3
        # compute the fusion edge based on the area of face
        w_edge = int(total_face_area ** 0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)

        if len(img.shape) == 2:  # upsample_img is gray image
            img = img[:, :, None]
        inv_soft_mask = inv_soft_mask[:, :, None]

        if len(img.shape) == 3 and img.shape[2] == 4:  # alpha channel
            alpha = img[:, :, 3:]
            img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * img[:, :, 0:3]
            img = np.concatenate((img, alpha), axis=2)
        else:
            img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * img

    img = img.astype(np.uint8)

    return img
