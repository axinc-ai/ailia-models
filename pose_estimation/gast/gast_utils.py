import numpy as np
import scipy.sparse as sp
import torch

__all__ = [
    'DataLoader',
    'coco_h36m',
    'get_joints_info',
    'normalize_screen_coordinates',
    'receptive_field',
]

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


class DataLoader(object):
    def __init__(
            self, poses_2d, pad=0, causal_shift=0,
            kps_left=None, kps_right=None):
        self.poses_2d = poses_2d
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.pad = pad
        self.causal_shift = causal_shift

    def next_epoch(self):
        for seq_2d in self.poses_2d:
            batch_2d = np.expand_dims(
                np.pad(seq_2d,
                       ((
                            self.pad + self.causal_shift,
                            self.pad - self.causal_shift
                        ),
                        (0, 0), (0, 0)),
                       'edge'),
                axis=0)

        batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
        batch_2d[1, :, :, 0] *= -1
        batch_2d[1, :, self.kps_left + self.kps_right] = \
            batch_2d[1, :, self.kps_right + self.kps_left]

        yield batch_2d.astype(np.float32)


def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2 * (
            keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3

    # half body: the joint of ankle and knee equal to hip
    # keypoints_h36m[:, [2, 3]] = keypoints_h36m[:, [1, 1]]
    # keypoints_h36m[:, [5, 6]] = keypoints_h36m[:, [4, 4]]

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    return keypoints_h36m, valid_frames


def get_joints_info(num_joints):
    # Body+toe keypoints
    if num_joints == 19:
        joints_left = [5, 6, 7, 8, 13, 14, 15]
        joints_right = [1, 2, 3, 4, 16, 17, 18]
    # Body keypoints
    else:
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

    keypoints_metadata = {
        'keypoints_symmetry': (joints_left, joints_right),
        'layout_name': 'Human3.6M',
        'num_joints': num_joints}

    return joints_left, joints_right, keypoints_metadata


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def receptive_field(filter_widths):
    """
    Return the total receptive field of this model as # of frames.
    """
    pad = [filter_widths[0] // 2]
    next_dilation = filter_widths[0]
    for i in range(1, len(filter_widths)):
        pad.append((filter_widths[i] - 1) * next_dilation // 2)
        next_dilation *= filter_widths[i]

    frames = 0
    for f in pad:
        frames += f
    return 1 + 2 * frames
