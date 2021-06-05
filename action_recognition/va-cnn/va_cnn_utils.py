import sys
import time

import numpy as np
import scipy
import scipy.misc

__all__ = [
    'torgb',
]


def center(rgb):
    rgb[:, :, 0] -= 110
    rgb[:, :, 1] -= 110
    rgb[:, :, 2] -= 110
    return rgb


def torgb(ske_joints, max_val, min_val):
    rgb = []
    maxmin = list()

    for ske_joint in ske_joints:
        zero_row = []

        for i in range(len(ske_joint)):
            if (ske_joint[i, :] == np.zeros((1, 150))).all():
                zero_row.append(i)
        ske_joint = np.delete(ske_joint, zero_row, axis=0)
        if (ske_joint[:, 0:75] == np.zeros((ske_joint.shape[0], 75))).all():
            ske_joint = np.delete(ske_joint, range(75), axis=1)
        elif (ske_joint[:, 75:150] == np.zeros((ske_joint.shape[0], 75))).all():
            ske_joint = np.delete(ske_joint, range(75, 150), axis=1)

        #### original rescale to 0-255
        ske_joint = 255 * (ske_joint - min_val) / (max_val - min_val)
        rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] // 3, 3))
        rgb_ske = scipy.misc.imresize(rgb_ske, (224, 224)).astype(np.float32)
        rgb_ske = center(rgb_ske)
        rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
        rgb_ske = np.transpose(rgb_ske, [2, 1, 0])
        rgb.append(rgb_ske)
        maxmin.append([max_val, min_val])

    return rgb, maxmin
