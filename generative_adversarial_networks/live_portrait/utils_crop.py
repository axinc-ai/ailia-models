from math import sin, cos, acos, degrees

import numpy as np
import cv2


DTYPE = np.float32


def _transform_img(img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=None):
    """conduct similarity or affine transformation to the image, do not do border operation!
    img:
    M: 2x3 matrix or 3x3 matrix
    dsize: target shape (width, height)
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(
            img,
            M[:2, :],
            dsize=_dsize,
            flags=flags,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )
    else:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def _transform_pts(pts, M):
    """conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]


def parse_pt2_from_pt106(pt106, use_lip=True):
    """
    parsing the 2 points according to the 106 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt_x(pts, use_lip=True):
    pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip)

    if not use_lip:
        # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]

    return pt2


def parse_rect_from_landmark(
    pts,
    scale=1.5,
    need_square=True,
    vx_ratio=0,
    vy_ratio=0,
    use_deg_flag=False,
    **kwargs,
):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get("use_lip", True))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=DTYPE)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=DTYPE)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale  # scale size
    center = (
        center0 + ux * center1[0] + uy * center1[1]
    )  # counterclockwise rotation, equivalent to M.T @ center1.T
    center = (
        center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    )  # considering the offset in vx and vy direction

    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle


def _estimate_similar_transform_from_pts(
    pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=-0.1, flag_do_rot=True, **kwargs
):
    """calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pts,
        scale=scale,
        vx_ratio=vx_ratio,
        vy_ratio=vy_ratio,
        use_lip=kwargs.get("use_lip", True),
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)  # center of dsize

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_INV = np.array(
            [
                [s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                [
                    -s * sintheta,
                    s * costheta,
                    tcy - s * (-sintheta * cx + costheta * cy),
                ],
            ],
            dtype=DTYPE,
        )
    else:
        M_INV = np.array(
            [
                [s, 0, tgt_center[0] - s * center[0]],
                [0, s, tgt_center[1] - s * center[1]],
            ],
            dtype=DTYPE,
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
    return M_INV, M[:2, ...]


def crop_image(img, pts: np.ndarray, dsize=224, scale=1.5, vy_ratio=-0.1):
    M_INV, _ = _estimate_similar_transform_from_pts(
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        flag_do_rot=True,
    )

    img_crop = _transform_img(img, M_INV, dsize)  # origin to crop
    pt_crop = _transform_pts(pts, M_INV)

    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    ret_dct = {
        "M_o2c": M_o2c,  # from the original image to the cropped image 3x3
        "M_c2o": M_c2o,  # from the cropped image to the original image 3x3
        "img_crop": img_crop,  # the cropped image
        "pt_crop": pt_crop,  # the landmarks of the cropped image
    }

    return ret_dct
