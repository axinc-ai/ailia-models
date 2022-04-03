import csv

import numpy as np

# annotation style of KITTI dataset
FIELDNAMES = [
    'type',
    'truncated',
    'occluded',
    'alpha',
    'xmin',
    'ymin',
    'xmax',
    'ymax',
    'dh',
    'dw',
    'dl',
    'lx',
    'ly',
    'lz',
    'ry'
]

# indices used for performing interpolation
# key->value: style->index arrays
interp_dict = {
    'bbox12': (
        np.array([
            1, 3, 5, 7,  # h direction
            1, 2, 3, 4,  # l direction
            1, 2, 5, 6]),  # w direction
        np.array([
            2, 4, 6, 8,
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


def csv_read_annot(file_path, pred=False):
    """
    Read instance attributes in the KITTI format. Instances not in the
    selected class will be ignored.

    A list of python dictionary is returned where each dictionary
    represents one instsance.
    """
    TYPE_ID_CONVERSION = {
        'Car': 0,
        'Cyclist': 1,
        'Pedestrian': 2,
    }

    fieldnames = FIELDNAMES
    if pred:
        fieldnames = FIELDNAMES + ['score']

    annotations = []
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            if row["type"] in ('Car',):
                annot_dict = {
                    "class": row["type"],
                    "label": TYPE_ID_CONVERSION[row["type"]],
                    "truncation": float(row["truncated"]),
                    "occlusion": float(row["occluded"]),
                    "alpha": float(row["alpha"]),
                    "dimensions": [
                        float(row['dl']),
                        float(row['dh']),
                        float(row['dw'])
                    ],
                    "locations": [
                        float(row['lx']),
                        float(row['ly']),
                        float(row['lz'])
                    ],
                    "rot_y": float(row["ry"]),
                    "bbox": [
                        float(row["xmin"]),
                        float(row["ymin"]),
                        float(row["xmax"]),
                        float(row["ymax"])
                    ]
                }
                if "score" in fieldnames:
                    annot_dict["score"] = float(row["score"])
                annotations.append(annot_dict)

    return annotations


def csv_read_calib(file_path):
    """
    Read camera projection matrix in the KITTI format.
    """
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P = row[1:]
                P = [float(i) for i in P]
                P = np.array(P, dtype=np.float32).reshape(3, 4)
                break

    return P


def augment_pose_vector(
        locs,
        rot_y,
        obj_class,
        dimension,
        augment,
        augment_times,
        std_rot=np.array([15., 50., 15.]) * np.pi / 180.,
        std_trans=np.array([0.2, 0.01, 0.2]), ):
    """
    Data augmentation used for training the lifter sub-model.

    std_rot: standard deviation of rotation around x, y and z axis
    std_trans: standard deviation of translation along x, y and z axis
    """
    aug_ids, aug_pose_vecs = [], []
    aug_ids.append((obj_class, dimension))
    # KITTI only annotates rotation around y-axis (yaw)
    pose_vec = np.concatenate([locs, np.array([0., rot_y, 0.])]).reshape(1, 6)
    aug_pose_vecs.append(pose_vec)
    if not augment:
        return aug_ids, aug_pose_vecs
    rots_random = np.random.randn(augment_times, 3) * std_rot.reshape(1, 3)
    # y-axis
    rots_random[:, 1] += rot_y
    trans_random = 1 + np.random.randn(augment_times, 3) * std_trans.reshape(1, 3)
    trans_random *= locs.reshape(1, 3)
    for i in range(augment_times):
        # augment 6DoF pose
        aug_ids.append((obj_class, dimension))
        pose_vec = np.concatenate([trans_random[i], rots_random[i]]).reshape(1, 6)
        aug_pose_vecs.append(pose_vec)

    return aug_ids, aug_pose_vecs


def interpolate(
        bbox_3d,
        style,
        interp_coef=[0.5],
        dimension=None):
    """
    Interpolate 3d points on a 3D bounding box with a specified style.
    """
    if dimension is not None:
        # size-encoded representation
        l = dimension[0]
        if l < 3.5:
            style += 'l'
        elif l < 4.5:
            style += 'h'
        else:
            style += 'w'
    pidx, cidx = interp_dict[style]
    parents, children = bbox_3d[:, pidx], bbox_3d[:, cidx]
    lines = children - parents
    new_joints = [(parents + interp_coef[i] * lines) for i in range(len(interp_coef))]
    return np.hstack([bbox_3d, np.hstack(new_joints)])


def construct_box_3d(l, h, w, interp_params):
    """
    Construct 3D bounding box corners in the canonical pose.
    """
    x_corners = [0.5 * l, l, l, l, l, 0, 0, 0, 0]
    y_corners = [0.5 * h, 0, h, 0, h, 0, h, 0, h]
    z_corners = [0.5 * w, w, w, 0, 0, w, w, 0, 0]
    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2
    corners_3d = np.array([x_corners, y_corners, z_corners])
    if interp_params['flag']:
        corners_3d = interpolate(
            corners_3d,
            interp_params['style'],
            interp_params['coef'],
        )

    return corners_3d


def get_cam_cord(cam_cord, shift, ids, pose_vecs, rot_xz=False):
    """
    Construct 3D bounding box corners in the camera coordinate system.
    """
    interp_params = {
        'flag': True, 'style': 'bbox12', 'coef': [0.332, 0.667]
    }

    # does not augment the dimension for now
    dims = ids[0][1]
    l, h, w = dims[0], dims[1], dims[2]
    corners_3d_fixed = construct_box_3d(l, h, w, interp_params)
    for pose_vec in pose_vecs:
        # translation
        locs = pose_vec[0, :3]
        rots = pose_vec[0, 3:]
        x, y, z = locs[0], locs[1], locs[2]  # bottom center of the labeled 3D box
        rx, ry, rz = rots[0], rots[1], rots[2]
        rot_maty = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        if rot_xz:
            # rotation. Only yaw angle is considered in KITTI dataset
            rot_matx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])
            rot_matz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])
            # TODO: correct here
            rot_mat = rot_matz @ rot_maty @ rot_matx
        else:
            rot_mat = rot_maty
        corners_3d = np.matmul(rot_mat, corners_3d_fixed)

        # translation
        corners_3d += np.array([x, y, z]).reshape([3, 1])
        camera_coordinates = corners_3d + shift
        cam_cord.append(camera_coordinates.T)

    return


def project_3d_to_2d(points, K):
    """
    Get 2D projection of 3D points in the camera coordinate system.
    """
    projected = K @ points.T
    projected[:2, :] /= projected[2, :]

    return projected


def get_inlier_indices(p_2d, threshold=0.3):
    """
    Get indices of instances that are visible 'enough'.
    """
    indices = []
    num_joints = p_2d[0].shape[0]
    for idx, kpts in enumerate(p_2d):
        if p_2d[idx][:, 2].sum() / num_joints >= threshold:
            indices.append(idx)

    return indices


def get_representation(p2d, p3d, in_rep, out_rep):
    """
    Get input-output representations based on 3d point cloud and its
    projected 2D screen coordinates.
    """
    # input representation
    if len(p2d) > 0:
        num_kpts = len(p2d[0])
    if in_rep == 'coordinates2d':
        input_list = [points.reshape(1, num_kpts, -1) for points in p2d]
    else:
        raise NotImplementedError('Undefined input representation.')

    # output representation
    if out_rep == 'R3d+T':
        # R3D stands for relative 3D shape, T stands for translation
        # center the camera coordinates to remove depth
        output_list = []
        for i in range(len(p3d)):
            # format: the root should be pre-computed as the first 3d point
            root = p3d[i][[0], :]
            relative_shape = p3d[i][1:, :] - root
            output = np.concatenate([root, relative_shape], axis=0)
            output_list.append(output.reshape(1, -1))
    else:
        raise NotImplementedError('undefined output representation.')

    return input_list, output_list


def _add_visibility(joints, img_width, img_height):
    """
    Compute binary visibility of projected 2D parts.
    """
    assert joints.shape[1] == 2
    visibility = np.ones((len(joints), 1))
    # predicate from upper left corner
    predicate1 = joints - np.array([[0., 0.]])
    predicate1 = (predicate1 > 0.).prod(axis=1)
    # predicate from lower right corner
    predicate2 = joints - np.array([[img_width, img_height]])
    predicate2 = (predicate2 < 0.).prod(axis=1)
    visibility[:, 0] *= predicate1 * predicate2

    return np.hstack([joints, visibility])


def get_2d_3d_pair(
        img,
        label_path,
        calib_path,
        in_rep='coordinates2d',
        out_rep='R3d+T',
        augment=False,
        augment_times=1,
        add_visibility=True,
        add_raw_bbox=False,  # add original bbox annotation from KITTI
        filter_outlier=False):
    anns = csv_read_annot(label_path)
    P = csv_read_calib(calib_path)

    # The intrinsics may vary slightly for different images
    # Yet one may convert them to a fixed one by applying a homography
    K = P[:, :3]

    shift = np.linalg.inv(K) @ P[:, 3].reshape(3, 1)

    # P containes intrinsics and extrinsics, I factorize P to K[I|K^-1t]
    # and use extrinsics to compute the camera coordinate
    # here the extrinsics represent the shift between current camera to
    # the reference grayscale camera
    # For more calibration details, refer to "Vision meets Robotics: The KITTI Dataset"
    camera_coordinates = []
    pose_vecs = []
    # id includes the class and size of the object
    ids = []
    bboxes = []
    for i, a in enumerate(anns):
        a = a.copy()
        obj_class = a["label"]
        dimension = a["dimensions"]
        locs = np.array(a["locations"])
        rot_y = np.array(a["rot_y"])
        if add_raw_bbox:
            bboxes.append(np.array(a["bbox"]).reshape(1, 4))

        aug_ids, aug_pose_vecs = augment_pose_vector(
            locs, rot_y, obj_class,
            dimension, augment, augment_times)
        get_cam_cord(
            camera_coordinates,
            shift,
            aug_ids,
            aug_pose_vecs)
        ids += aug_ids
        pose_vecs += aug_pose_vecs

    num_instances = len(camera_coordinates)

    # get 2D projections
    if len(camera_coordinates) != 0:
        camera_coordinates = np.vstack(camera_coordinates)
        projected = project_3d_to_2d(camera_coordinates, K)[:2, :].T
        # target is camera coordinates
        p_2d = np.split(projected, num_instances, axis=0)
        p_3d = np.split(camera_coordinates, num_instances, axis=0)
        # set visibility to 0 if the projected keypoints lie out of the image plane
        if add_visibility:
            height, width = img.shape[:2]
            for idx, joints in enumerate(p_2d):
                p_2d[idx] = _add_visibility(joints, width, height)

        # filter out the instances that lie outside of the image
        if filter_outlier:
            indices = get_inlier_indices(p_2d)
            p_2d = [p_2d[idx] for idx in indices]
            p_3d = [p_3d[idx] for idx in indices]
            if add_raw_bbox:
                bboxes = [bboxes[idx] for idx in indices]
        list_2d, list_3d = get_representation(p_2d, p_3d, in_rep, out_rep)
    else:
        list_2d, list_3d, ids, pose_vecs = [], [], [], []

    ret = list_2d, list_3d, ids, pose_vecs, K, anns
    if add_raw_bbox:
        ret = ret + (bboxes,)

    return ret
