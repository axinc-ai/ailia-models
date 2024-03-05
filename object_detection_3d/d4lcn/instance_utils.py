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
    'ry',
    'score'
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


def read_annot(file):
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

    annotations = []
    reader = csv.DictReader(file, delimiter=' ', fieldnames=fieldnames)
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


def read_calib_file(file_path):
    """
    Read camera projection matrix in the KITTI format.
    """
    p2 = np.zeros([4, 4], dtype=float)

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                _p2 = row[1:]
                _p2 = [float(i) for i in _p2]
                _p2 = np.array(_p2, dtype=np.float32).reshape(3, 4)

                p2[:3, :4] = _p2
                p2[3, 3] = 1

                break

    return p2


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


def get_2d(
        anns,
        p2,
        augment=False,
        augment_times=1):
    # The intrinsics may vary slightly for different images
    # Yet one may convert them to a fixed one by applying a homography
    K = p2[:, :3]

    shift = np.linalg.inv(K) @ p2[:, 3].reshape(3, 1)

    # P containes intrinsics and extrinsics, I factorize P to K[I|K^-1t]
    # and use extrinsics to compute the camera coordinate
    # here the extrinsics represent the shift between current camera to
    # the reference grayscale camera
    # For more calibration details, refer to "Vision meets Robotics: The KITTI Dataset"
    camera_coordinates = []
    pose_vecs = []
    # id includes the class and size of the object
    ids = []
    for i, a in enumerate(anns):
        a = a.copy()
        obj_class = a["label"]
        dimension = a["dimensions"]
        locs = np.array(a["locations"])
        rot_y = np.array(a["rot_y"])

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
        kpts_2d = np.split(projected, num_instances, axis=0)
    else:
        kpts_2d = []

    return kpts_2d
