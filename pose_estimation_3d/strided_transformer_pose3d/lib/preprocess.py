import json
import numpy as np
import os

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


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
    keypoints_h36m[:, 7, 0] += 2*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    
    return keypoints_h36m, valid_frames


def h36m_coco_format(keypoints, scores):
    assert len(keypoints.shape) == 4 and len(scores.shape) == 3

    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.:
            kpts, valid_frame = coco_h36m(kpts)
            h36m_kpts.append(kpts)
            valid_frames.append(valid_frame)

            new_score[:, h36m_coco_order] = score[:, coco_order]
            new_score[:, 0] = np.mean(score[:, [11, 12]], axis=1, dtype=np.float32)
            new_score[:, 8] = np.mean(score[:, [5, 6]], axis=1, dtype=np.float32)
            new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
            new_score[:, 10] = np.mean(score[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

            h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts

