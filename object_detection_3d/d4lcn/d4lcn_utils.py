import math

import numpy as np


def bbox_transform_inv(boxes, deltas, means=None, stds=None):
    """
    Compute the bbox target transforms in 3D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if stds is not None:
        dx *= stds[0]
        dy *= stds[1]
        dw *= stds[2]
        dh *= stds[3]

    if means is not None:
        dx += means[0]
        dy += means[1]
        dw += means[2]
        dh += means[3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros(deltas.shape)

    # x1, y1, x2, y2
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def convertAlpha2Rot(alpha, z3d, x3d):
    ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi

    while ry3d > math.pi:
        ry3d -= math.pi * 2
    while ry3d < (-math.pi):
        ry3d += math.pi * 2

    return ry3d


def convertRot2Alpha(ry3d, z3d, x3d):
    alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi

    while alpha > math.pi:
        alpha -= math.pi * 2
    while alpha < (-math.pi):
        alpha += math.pi * 2

    return alpha


def project_3d(
        p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate object coordinate to camera coordinate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]  # normalize dim 3 -> 2, image coordinate

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d  # 2d corners in image coordinate and 3d corners in camera coordinate
    else:
        return verts3d


def test_projection(p2, p2_inv, box_2d, cx, cy, z, w3d, h3d, l3d, rotY):
    """
    Tests the consistency of a 3D projection compared to a 2D box
    """

    x = box_2d[0]
    y = box_2d[1]
    x2 = x + box_2d[2] - 1
    y2 = y + box_2d[3] - 1

    coord3d = p2_inv.dot(np.array([cx * z, cy * z, z, 1]))  # camera coordinate

    cx3d = coord3d[0]
    cy3d = coord3d[1]
    cz3d = coord3d[2]

    # put back on ground first
    # cy3d += h3d/2

    # re-compute the 2D box using 3D (finally, avoids clipped boxes)
    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    invalid = np.any(corners_3d[2, :] <= 0)

    x_new = min(verts3d[:, 0])
    y_new = min(verts3d[:, 1])
    x2_new = max(verts3d[:, 0])
    y2_new = max(verts3d[:, 1])

    b1 = np.array([x, y, x2, y2])[np.newaxis, :]
    b2 = np.array([x_new, y_new, x2_new, y2_new])[np.newaxis, :]

    # ol = iou(b1, b2)[0][0]
    ol = -(np.abs(x - x_new) + np.abs(y - y_new) + np.abs(x2 - x2_new) + np.abs(y2 - y2_new))  # L1 norm

    return ol, verts3d, b2, invalid


# box_2d = [x1, y1, width, height], only use r in Exp. x2d, y2d, z2d -> 2d center, depth. rY is converted from alpha.
def hill_climb(
        p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d,
        step_z_init=0, step_r_init=0, z_lim=0, r_lim=0,
        min_ol_dif=0.0):
    step_z = step_z_init
    step_r = step_r_init

    ol_best, verts_best, _, invalid = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d)

    if invalid:
        return z2d, ry3d, verts_best

    # attempt to fit z/rot more properly
    while (step_z > z_lim or step_r > r_lim):
        if step_z > z_lim:
            ol_neg, verts_neg, _, invalid_neg = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d - step_z,
                w3d, h3d, l3d, ry3d)
            ol_pos, verts_pos, _, invalid_pos = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d + step_z,
                w3d, h3d, l3d, ry3d)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_z = step_z * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                z2d += step_z
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                z2d -= step_z
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_z = step_z * 0.5

        if step_r > r_lim:
            ol_neg, verts_neg, _, invalid_neg = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d,
                w3d, h3d, l3d, ry3d - step_r)
            ol_pos, verts_pos, _, invalid_pos = test_projection(
                p2, p2_inv, box_2d, x2d, y2d, z2d,
                w3d, h3d, l3d, ry3d + step_r)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_r = step_r * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                ry3d += step_r
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                ry3d -= step_r
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_r = step_r * 0.5

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return z2d, ry3d, verts_best
