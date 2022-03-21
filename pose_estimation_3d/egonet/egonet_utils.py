import numpy as np
import cv2

SIZE = 200.0


def get_affine_transform(
        center,
        scale,
        rot,
        output_size,
        shift=np.array([0, 0], dtype=np.float32),
        inv=0):
    """
    Estimate an affine transformation given crop parameters (center, scale and
    rotation) and output resolution.
    """
    if isinstance(scale, list):
        scale = np.array(scale)
    if isinstance(center, list):
        center = np.array(center)

    scale_tmp = scale * SIZE
    src_w = scale_tmp[0]
    dst_h, dst_w = output_size

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def resize_bbox(left, top, right, bottom, target_ar=1.):
    """
    Resize a bounding box to pre-defined aspect ratio.
    """
    width = right - left
    height = bottom - top
    aspect_ratio = height / width
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    if aspect_ratio > target_ar:
        new_width = height * (1 / target_ar)
        new_left = center_x - 0.5 * new_width
        new_right = center_x + 0.5 * new_width
        new_top = top
        new_bottom = bottom
    else:
        new_height = width * target_ar
        new_left = left
        new_right = right
        new_top = center_y - 0.5 * new_height
        new_bottom = center_y + 0.5 * new_height

    d = {
        'bbox': [new_left, new_top, new_right, new_bottom],
        'c': np.array([center_x, center_y]),
        's': np.array([(new_right - new_left) / SIZE, (new_bottom - new_top) / SIZE])
    }
    return d


def enlarge_bbox(left, top, right, bottom, enlarge):
    """
    Enlarge a bounding box.
    """
    width = right - left
    height = bottom - top
    new_width = width * enlarge[0]
    new_height = height * enlarge[1]
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    new_left = center_x - 0.5 * new_width
    new_right = center_x + 0.5 * new_width
    new_top = center_y - 0.5 * new_height
    new_bottom = center_y + 0.5 * new_height

    return [new_left, new_top, new_right, new_bottom]


def modify_bbox(bbox, target_ar, enlarge=1.1):
    """
    Modify a bounding box by enlarging/resizing.
    """
    lbbox = enlarge_bbox(bbox[0], bbox[1], bbox[2], bbox[3], [enlarge, enlarge])
    ret = resize_bbox(lbbox[0], lbbox[1], lbbox[2], lbbox[3], target_ar=target_ar)

    return ret


def resize_crop(crop_size, target_ar=None):
    """
    Resize a crop size to a pre-defined aspect ratio.
    """
    if target_ar is None:
        return crop_size

    width = crop_size[0]
    height = crop_size[1]
    aspect_ratio = height / width
    if aspect_ratio > target_ar:
        new_width = height * (1 / target_ar)
        new_height = height
    else:
        new_height = width * target_ar
        new_width = width

    return [new_width, new_height]


def cs2bbox(center, size):
    """
    Convert center/scale to a bounding box annotation.
    """
    x1 = center[0] - size[0]
    y1 = center[1] - size[1]
    x2 = center[0] + size[0]
    y2 = center[1] + size[1]

    return [x1, y1, x2, y2]


def kpts2cs(
        keypoints,
        enlarge=1.1,
        method='boundary',
        target_ar=None,
        use_visibility=True):
    """
    Convert instance screen coordinates to cropping center and size

    keypoints of shape [n_joints, 2/3]
    """
    if keypoints.shape[1] == 2:
        visible_keypoints = keypoints
        vis_rate = 1.0
    elif keypoints.shape[1] == 3 and use_visibility:
        visible_indices = keypoints[:, 2].nonzero()[0]
        visible_keypoints = keypoints[visible_indices, :2]
        vis_rate = len(visible_keypoints) / len(keypoints)
    else:
        visible_keypoints = keypoints[:, :2]
        visible_indices = np.array(range(len(keypoints)))
        vis_rate = 1.0

    if method == 'centroid':
        center = np.ceil(visible_keypoints.mean(axis=0, keepdims=True))
        dif = np.abs(visible_keypoints - center).max(axis=0, keepdims=True)
        crop_size = np.ceil(dif * enlarge).squeeze()
        center = center.squeeze()
    elif method == 'boundary':
        left_top = visible_keypoints.min(axis=0, keepdims=True)
        right_bottom = visible_keypoints.max(axis=0, keepdims=True)
        center = ((left_top + right_bottom) / 2).squeeze()
        crop_size = ((right_bottom - left_top) * enlarge / 2).squeeze()
    else:
        raise NotImplementedError

    # resize the bounding box to a specified aspect ratio
    crop_size = resize_crop(crop_size, target_ar)
    x1, y1, x2, y2 = cs2bbox(center, crop_size)

    new_origin = np.array([[x1, y1]], dtype=keypoints.dtype)
    new_keypoints = keypoints.copy()
    if keypoints.shape[1] == 2:
        new_keypoints = visible_keypoints - new_origin
    elif keypoints.shape[1] == 3:
        new_keypoints[visible_indices, :2] = visible_keypoints - new_origin

    return center, crop_size, new_keypoints, vis_rate
