import math

import numpy as np
import numba


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
