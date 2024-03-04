import os
from io import BytesIO
from math import cos, atan2, asin, sqrt
import ctypes

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
landmarks
"""


def draw_landmarks(img, pts, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    dense_flag = kwargs.get('dense_flag')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if dense_flag:
            plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=0.4, color='c', alpha=0.7)
        else:
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot(
                [pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(
                    pts[i][0, l:r], pts[i][1, l:r],
                    marker='o', linestyle='None',
                    markersize=markersize, color=color,
                    markeredgecolor=markeredgecolor, alpha=alpha)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    img = cv2.imdecode(img, 1)

    return img


"""
pose
"""


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)

    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]

    return P, pose


def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    return point_3d


def plot_pose_box(img, P, ver, color=(40, 255, 0), line_width=2):
    """ Draw a 3D box as annotation of pose.
    Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        img: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (2, 68) or (3, 68)
    """
    llength = calc_hypotenuse(ver)
    point_3d = build_camera_box(llength)
    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(ver[:2, :27], 1)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return img


def viz_pose(img, param_lst, ver_lst):
    for param, ver in zip(param_lst, ver_lst):
        P, pose = calc_pose(param)
        img = plot_pose_box(img, P, ver)
        print(f'yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')

    return img


"""
render
"""


class TrianglesMeshRender:
    def __init__(
            self,
            light=(0, 0, 5),
            direction=(0.6, 0.6, 0.6),
            ambient=(0.3, 0.3, 0.3)
    ):
        clibs = "asset/render.so"
        if not os.path.exists(clibs):
            raise Exception(
                f'{clibs} not found, please build it first, by run '
                f'"gcc -shared -Wall -O3 render.c -o render.so -fPIC" in asset directory')

        self._clibs = ctypes.CDLL(clibs)

        self._light = np.array(light, dtype=np.float32)
        self._light = np.ctypeslib.as_ctypes(self._light)

        self._direction = np.array(direction, dtype=np.float32)
        self._direction = np.ctypeslib.as_ctypes(self._direction)

        self._ambient = np.array(ambient, dtype=np.float32)
        self._ambient = np.ctypeslib.as_ctypes(self._ambient)

    def __call__(self, vertices, triangles, bg):
        tri_nums = triangles.shape[0]
        ver_nums = vertices.shape[0]
        bg_shape = bg.shape

        triangles = np.ctypeslib.as_ctypes(3 * triangles)  # Attention
        vertices = np.ctypeslib.as_ctypes(vertices.copy(order='C'))
        bg = np.ctypeslib.as_ctypes(bg)
        self._clibs._render(
            triangles, tri_nums,
            self._light, self._direction, self._ambient,
            vertices, ver_nums,
            bg, bg_shape[0], bg_shape[1]
        )


def render(img, ver_lst, tri, alpha=0.6, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    render_app = TrianglesMeshRender()

    for ver_ in ver_lst:
        ver = np.ascontiguousarray(ver_.T)  # transpose
        render_app(ver, tri, bg=overlap)

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    return res
