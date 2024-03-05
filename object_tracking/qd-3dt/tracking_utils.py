import math

import numpy as np
import numba


def yaw2alpha(rot_y, x_loc, z_loc):
    """
    Get alpha by rotation_y - theta
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    torch_pi = np.array([np.pi])
    alpha = rot_y - np.arctan2(x_loc, z_loc)
    alpha = (alpha + torch_pi) % (2 * torch_pi) - torch_pi
    return alpha


def alpha2yaw(alpha, x_loc, z_loc):
    """
    Get rotation_y by alpha + theta
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    pi = np.array([np.pi])
    rot_y = alpha + np.arctan2(x_loc, z_loc)
    rot_y = (rot_y + pi) % (2 * pi) - pi

    return rot_y


def imagetocamera(points, depths, projection):
    """
    points: (N, 2), N points on X-Y image plane
    depths: (N,), N depth values for points
    projection: (3, 4), projection matrix

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    corners = np.matmul(
        np.concatenate([
            points, np.ones((points.shape[0], 1))
        ], axis=1),
        np.linalg.inv(projection[:, 0:3]).T)

    corners_cam = corners * depths.reshape(-1, 1)

    return corners_cam


def cameratoimage(corners, projection, invalid_value=-1000):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera plane
    projection: (3, 4), projection matrix

    points: (N, 2), N points on X-Y image plane
    """
    assert corners.shape[1] == 3, "Shape ({}) not fit".format(corners.shape)

    points = np.matmul(np.concatenate([
        corners, np.ones((corners.shape[0], 1))
    ], axis=1), projection.T)

    # [x, y, z] -> [x/z, y/z]
    mask = points[:, 2:3] > 0
    points_img = (points[:, :2] / points[:, 2:3]) * mask + invalid_value * np.logical_not(mask)

    return points_img


def cameratoworld(corners, position, rotation):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    corners_global = np.matmul(corners, rotation.T) + position[None]
    return corners_global


def worldtocamera(corners_global, position, rotation):
    """
    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert corners_global.shape[1] == 3, ("Shape ({}) not fit".format(
        corners_global.shape))
    corners = np.matmul(corners_global - position[None], rotation)
    return corners


def computeboxes(roty, dim, loc):
    '''Get 3D bbox vertex in camera coordinates
    Input:
        roty: (1,), object orientation, -pi ~ pi
        box_dim: a tuple of (h, w, l)
        loc: (3,), box 3D center
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    roty = roty[0]
    R = np.array([
        [+np.cos(roty), 0, +np.sin(roty)], [0, 1, 0],
        [-np.sin(roty), 0, +np.cos(roty)]
    ])
    corners = get_vertex(dim)
    corners = corners.dot(R.T) + loc
    return corners


def get_vertex(box_dim):
    '''Get 3D bbox vertex (used for the upper volume iou calculation)
    Input:
        box_dim: a tuple of (h, w, l)
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    h, w, l = box_dim
    corners = np.array([
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
        [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    ])
    return corners.T


def get_3d_bbox_vertex(cam_calib, cam_pose, points3d, cam_near_clip=0.15):
    '''Get 3D bbox vertex in camera coordinates
    Input:
        cam_calib: (3, 4), projection matrix
        cam_pose: a class with position, rotation of the frame
            rotation:  (3, 3), rotation along camera coordinates
            position:  (3), translation of world coordinates
        points3d: (8, 3), box 3D center in camera coordinates
        cam_near_clip: in meter, distance to the near plane
    Output:
        points: numpy array of shape (8, 2) for bbox in image coordinates
    '''
    lineorder = np.array(
        [
            [1, 2, 6, 5],  # front face
            [2, 3, 7, 6],  # left face
            [3, 4, 8, 7],  # back face
            [4, 1, 5, 8],
            [1, 6, 5, 2]
        ],
        dtype=np.int32) - 1  # right

    points = []

    # In camera coordinates
    cam_dir = np.array([0, 0, 1])
    center_pt = cam_dir * cam_near_clip

    for i in range(len(lineorder)):
        for j in range(4):
            p1 = points3d[lineorder[i, j]].copy()
            p2 = points3d[lineorder[i, (j + 1) % 4]].copy()

            before1 = is_before_clip_plane_camera(
                p1[np.newaxis], cam_near_clip)[0]
            before2 = is_before_clip_plane_camera(
                p2[np.newaxis], cam_near_clip)[0]

            inter = get_intersect_point(center_pt, cam_dir, p1, p2)

            if not (before1 or before2):
                # print("Not before 1 or 2")
                continue
            elif before1 and before2:
                # print("Both 1 and 2")
                vp1 = p1
                vp2 = p2
            elif before1 and not before2:
                # print("before 1 not 2")
                vp1 = p1
                vp2 = inter
            elif before2 and not before1:
                # print("before 2 not 1")
                vp1 = inter
                vp2 = p2

            cp1 = cameratoimage(vp1[np.newaxis], cam_calib)[0]
            cp2 = cameratoimage(vp2[np.newaxis], cam_calib)[0]
            points.append((cp1, cp2))
    return points


@numba.jit()
def alpha2rot_y(alpha, x, FOCAL_LENGTH):
    """
    Get rotation_y by alpha + theta
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x, FOCAL_LENGTH)
    rot_y = (rot_y + np.pi) % (2 * np.pi) - np.pi
    return rot_y


@numba.jit(nopython=True, nogil=True)
def rot_axis(angle, axis):
    # RX = np.array([ [1,             0,              0],
    #                 [0, np.cos(gamma), -np.sin(gamma)],
    #                 [0, np.sin(gamma),  np.cos(gamma)]])
    #
    # RY = np.array([ [ np.cos(beta), 0, np.sin(beta)],
    #                 [            0, 1,            0],
    #                 [-np.sin(beta), 0, np.cos(beta)]])
    #
    # RZ = np.array([ [np.cos(alpha), -np.sin(alpha), 0],
    #                 [np.sin(alpha),  np.cos(alpha), 0],
    #                 [            0,              0, 1]])
    cg = np.cos(angle)
    sg = np.sin(angle)
    if axis == 0:  # X
        v = [0, 4, 5, 7, 8]
    elif axis == 1:  # Y
        v = [4, 0, 6, 2, 8]
    else:  # Z
        v = [8, 0, 1, 3, 4]
    RX = np.zeros(9, dtype=numba.float64)
    RX[v[0]] = 1.0
    RX[v[1]] = cg
    RX[v[2]] = -sg
    RX[v[3]] = sg
    RX[v[4]] = cg
    return RX.reshape(3, 3)


# Same as angle2rot from kio_slim
def angle2rot(rotation, inverse=False):
    return rotate(np.eye(3), rotation, inverse=inverse)


@numba.jit(nopython=True, nogil=True)
def rotate(vector, angle, inverse=False):
    """
    Rotation of x, y, z axis
    Forward rotate order: Z, Y, X
    Inverse rotate order: X^T, Y^T,Z^T
    Input:
        vector: vector in 3D coordinates
        angle: rotation along X, Y, Z (raw data from GTA)
    Output:
        out: rotated vector
    """
    gamma, beta, alpha = angle[0], angle[1], angle[2]

    # Rotation matrices around the X (gamma), Y (beta), and Z (alpha) axis
    RX = rot_axis(gamma, 0)
    RY = rot_axis(beta, 1)
    RZ = rot_axis(alpha, 2)

    # Composed rotation matrix with (RX, RY, RZ)
    if inverse:
        return np.dot(np.dot(np.dot(RX.T, RY.T), RZ.T), vector)
    else:
        return np.dot(np.dot(np.dot(RZ, RY), RX), vector)


@numba.jit(nopython=True)
def get_intersect_point(center_pt, cam_dir, vertex1, vertex2):
    # get the intersection point of two 3D points and a plane
    c1 = center_pt[0]
    c2 = center_pt[1]
    c3 = center_pt[2]
    a1 = cam_dir[0]
    a2 = cam_dir[1]
    a3 = cam_dir[2]
    x1 = vertex1[0]
    y1 = vertex1[1]
    z1 = vertex1[2]
    x2 = vertex2[0]
    y2 = vertex2[1]
    z2 = vertex2[2]

    k_up = abs(a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3))
    k_down = abs(a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2))
    if k_up > k_down:
        k = 1
    else:
        k = k_up / k_down
    inter_point = (1 - k) * vertex1 + k * vertex2
    return inter_point


def is_before_clip_plane_camera(points_camera, cam_near_clip=0.15):
    """
    points_camera: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    cam_near_clip: scalar, the near projection plane

    is_before: bool, is the point locate before the near clip plane
    """
    return points_camera[:, 2] > cam_near_clip


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) \
         - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) \
         + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) \
         - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) \
         + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw
