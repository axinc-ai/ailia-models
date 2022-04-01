from copy import deepcopy
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from points_utils import plot_2d_bbox, plot_3d_bbox
from points_utils import plot_scene_3dbox, draw_pose_vecs

SIZE = 200.0

# indices used for performing interpolation
# key->value: style->index arrays
interp_dict = {
    'bbox12': (
        np.array([1, 3, 5, 7,  # h direction
                  1, 2, 3, 4,  # l direction
                  1, 2, 5, 6]),  # w direction
        np.array([2, 4, 6, 8,
                  5, 6, 7, 8,
                  3, 4, 7, 8])
    ),
    'bbox12l': (
        np.array([1, 2, 3, 4, ]),  # w direction
        np.array([5, 6, 7, 8])
    ),
    'bbox12h': (
        np.array([1, 3, 5, 7]),  # w direction
        np.array([2, 4, 6, 8])
    ),
    'bbox12w': (
        np.array([1, 2, 5, 6]),  # w direction
        np.array([3, 4, 7, 8])
    ),
}


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


def affine_transform_modified(pts, t):
    """
    Apply affine transformation with homogeneous coordinates.
    """
    # pts of shape [n, 2]
    new_pts = np.hstack([pts, np.ones((len(pts), 1))]).T
    new_pts = t @ new_pts

    return new_pts[:2, :].T


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


def get_observation_angle_trans(euler_angles, translations):
    """
    Convert orientation in camera coordinate into local coordinate system
    utilizing known object location (translation)
    """
    alphas = euler_angles[:, 1].copy()
    for idx in range(len(euler_angles)):
        ry3d = euler_angles[idx][1]  # orientation in the camera coordinate system
        x3d, z3d = translations[idx][0], translations[idx][2]
        alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
        # alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi
        while alpha > math.pi: alpha -= math.pi * 2
        while alpha < (-math.pi): alpha += math.pi * 2
        alphas[idx] = alpha

    return alphas


def get_observation_angle_proj(euler_angles, kpts, K):
    """
    Convert orientation in camera coordinate into local coordinate system
    utilizing the projection of object on the image plane
    """
    f = K[0, 0]
    cx = K[0, 2]
    kpts_x = [kpts[i][0, 0] for i in range(len(kpts))]
    alphas = euler_angles[:, 1].copy()
    for idx in range(len(euler_angles)):
        ry3d = euler_angles[idx][1]  # orientation in the camera coordinate system
        x3d, z3d = kpts_x[idx] - cx, f
        alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
        # alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi
        while alpha > math.pi: alpha -= math.pi * 2
        while alpha < (-math.pi): alpha += math.pi * 2
        alphas[idx] = alpha

    return alphas


def get_template(prediction, interp_coef=[0.332, 0.667]):
    """
    Construct a template 3D cuboid at canonical pose. The 3D cuboid is
    represented as part coordinates in the camera coordinate system.
    """
    parents = prediction[interp_dict['bbox12'][0] - 1]
    children = prediction[interp_dict['bbox12'][1] - 1]
    lines = parents - children
    lines = np.sqrt(np.sum(lines ** 2, axis=1))

    # averaged over the four parallel line segments
    h, l, w = np.sum(lines[:4]) / 4, np.sum(lines[4:8]) / 4, np.sum(lines[8:]) / 4
    x_corners = [l, l, l, l, 0, 0, 0, 0]
    y_corners = [0, h, 0, h, 0, h, 0, h]
    z_corners = [w, w, 0, 0, w, w, 0, 0]
    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2
    corners_3d = np.array([x_corners, y_corners, z_corners])
    if len(prediction) == 32:
        pidx, cidx = interp_dict['bbox12']
        parents, children = corners_3d[:, pidx - 1], corners_3d[:, cidx - 1]
        lines = children - parents
        new_joints = [(parents + interp_coef[i] * lines) for i in range(len(interp_coef))]
        corners_3d = np.hstack([corners_3d, np.hstack(new_joints)])

    return corners_3d


def compute_rigid_transform(X, Y, W=None):
    """
    A least-sqaure estimate of rigid transformation by SVD.

    Reference: https://content.sakai.rutgers.edu/access/content/group/
    7bee3f05-9013-4fc2-8743-3c5078742791/material/svd_ls_rotation.pdf

    X, Y: [d, N] N data points of dimention d
    W: [N, ] optional weight (importance) matrix for N data points
    """

    # find mean column wise
    centroid_X = np.mean(X, axis=1, keepdims=True)
    centroid_Y = np.mean(Y, axis=1, keepdims=True)
    # subtract mean
    Xm = X - centroid_X
    Ym = Y - centroid_Y
    if W is None:
        H = Xm @ Ym.T
    else:
        W = np.diag(W) if len(W.shape) == 1 else W
        H = Xm @ W @ Ym.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # the global minimizer with a orthogonal transformation is not possible
        # the next best transformation is chosen
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_X + centroid_Y

    return R, t


def kpts_to_euler(template, prediction):
    """
    Convert the predicted cuboid representation to euler angles.
    """
    # estimate roll, pitch, yaw of the prediction by comparing with a
    # reference bounding box
    # prediction and template of shape [3, N_points]
    R, T = compute_rigid_transform(template, prediction)
    # in the order of yaw, pitch and roll
    angles = Rotation.from_matrix(R).as_euler('yxz', degrees=False)
    # re-order in the order of x, y and z
    angles = angles[[1, 0, 2]]

    return angles, T


def get_6d_rep(predictions):
    """
    Get the 6DoF representation of a 3D prediction.
    """
    predictions = predictions.reshape(len(predictions), -1, 3)
    all_angles = []
    for instance_idx in range(len(predictions)):
        prediction = predictions[instance_idx]
        # templates are 3D boxes with no rotation
        # the prediction is estimated as the rotation between prediction and template
        template = get_template(prediction)
        instance_angle, instance_trans = kpts_to_euler(template, prediction.T)
        all_angles.append(instance_angle.reshape(1, 3))

    angles = np.concatenate(all_angles)
    # the first point is the predicted point center
    translation = predictions[:, 0, :]

    return angles, translation


def get_instance_str(dic):
    """
    Produce KITTI style prediction string for one instance.
    """
    string = ""
    string += dic['class'] + " "
    string += "{:.1f} ".format(dic['truncation'])
    string += "{:.1f} ".format(dic['occlusion'])
    string += "{:.6f} ".format(dic['alpha'])
    string += "{:.6f} {:.6f} {:.6f} {:.6f} ".format(dic['bbox'][0], dic['bbox'][1], dic['bbox'][2], dic['bbox'][3])
    string += "{:.6f} {:.6f} {:.6f} ".format(dic['dimensions'][1], dic['dimensions'][2], dic['dimensions'][0])
    string += "{:.6f} {:.6f} {:.6f} ".format(dic['locations'][0], dic['locations'][1], dic['locations'][2])
    string += "{:.6f} ".format(dic['rot_y'])
    if 'score' in dic:
        string += "{:.8f} ".format(dic['score'])
    else:
        string += "{:.8f} ".format(1.0)

    return string


def get_pred_str(record):
    """
    Produce KITTI style prediction string for a record dictionary.
    """
    # replace the rotation predictions of input bounding boxes
    updated_txt = deepcopy(record['raw_txt_format'])
    for instance_id in range(len(record['euler_angles'])):
        updated_txt[instance_id]['rot_y'] = record['euler_angles'][instance_id, 1]
        updated_txt[instance_id]['alpha'] = record['alphas'][instance_id]

    pred_str = ""
    angles = record['euler_angles']
    for instance_id in range(len(angles)):
        # format a string for submission
        tempt_str = get_instance_str(updated_txt[instance_id])
        if instance_id != len(angles) - 1:
            tempt_str += '\n'
        pred_str += tempt_str

    return pred_str


def plot_2d_objects(
        img, record,
        color_dict={
            'bbox_2d': 'r',
            'bbox_3d': 'r',
            'kpts': ['rx', 'b']
        },
        ax=None):
    if ax:
        fig = None
    else:
        fig = plt.figure(figsize=(11.3, 9))
        ax = plt.subplot(111)

        height, width, _ = img.shape
        ax.imshow(img)
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.invert_yaxis()

    for idx in range(len(record['kpts_2d_pred'])):
        kpts = record['kpts_2d_pred'][idx].reshape(-1, 2)
        bbox = record['bbox_resize'][idx]
        plot_2d_bbox(ax, bbox, color_dict['bbox_2d'])
        # predicted key-points
        ax.plot(kpts[:, 0], kpts[:, 1], color_dict['kpts'])

    if 'kpts_2d_gt' in record:
        # plot ground truth 2D screen coordinates
        for idx, kpts_gt in enumerate(record['kpts_2d_gt']):
            kpts_gt = kpts_gt.reshape(-1, 3)
            plot_3d_bbox(ax, kpts_gt[1:, :2], color='g', linestyle='-.')

    if 'arrow' in record:
        for idx in range(len(record['arrow'])):
            start = record['arrow'][idx][:, 0]
            end = record['arrow'][idx][:, 1]
            x, y = start
            dx, dy = end - start
            ax.arrow(x, y, dx, dy, color='r', lw=4, head_width=5, alpha=0.5)

    if fig:
        fig.gca().set_axis_off()
        fig.subplots_adjust(
            top=1, bottom=0, right=1, left=0,
            hspace=0, wspace=0)
        fig.gca().xaxis.set_major_locator(plt.NullLocator())
        fig.gca().yaxis.set_major_locator(plt.NullLocator())

    return fig, ax


def plot_3d_objects(prediction, target, pose_vecs_gt, record, color, ax=None):
    if target is not None:
        p3d_gt = target.reshape(len(target), -1, 3)
    else:
        p3d_gt = None
    p3d_pred = prediction.reshape(len(prediction), -1, 3)

    if p3d_gt is not None:
        # use ground truth translation for visualization
        p3d_pred = np.concatenate([p3d_gt[:, [0], :], p3d_pred], axis=1)
    elif "kpts_3d" in record:
        # use predicted translation for visualization
        p3d_pred = np.concatenate([record['kpts_3d'][:, [0], :], p3d_pred], axis=1)
    else:
        raise NotImplementedError

    if ax:
        fig = None
    else:
        fig = plt.figure()
        ax = plt.subplot(111, projection='3d')
        
        ax.set_title("GT: black w/o Ego-Net: magenta w/ Ego-Net: red/yellow")

    # plotting a set of 3D boxes
    ax = plot_scene_3dbox(ax, p3d_pred, p3d_gt, color=color)
    # draw pose angle predictions
    translation = p3d_pred[:, 0, :]
    pose_vecs_pred = np.concatenate([translation, record['euler_angles']], axis=1)
    draw_pose_vecs(ax, pose_vecs_pred, color=color)
    draw_pose_vecs(ax, pose_vecs_gt)

    if p3d_gt is None:
        # plot input 3D bounding boxes before using Ego-Net
        kpts_3d = record['kpts_3d']
        plot_scene_3dbox(ax, kpts_3d, color='m')
        # pose_vecs_before = np.zeros((len(kpts_3d), 6))
        # for idx in range(len(pose_vecs_before)):
            # pose_vecs_before[idx][0:3] = record['raw_txt_format'][idx]['locations']
            # pose_vecs_before[idx][4] = record['raw_txt_format'][idx]['rot_y']
        # draw_pose_vecs(ax, pose_vecs_before, color='m')

    if fig:
        fig.gca().set_axis_off()
        fig.subplots_adjust(
            top=1, bottom=0, right=1, left=0,
            hspace=0, wspace=0)

    return fig, ax
