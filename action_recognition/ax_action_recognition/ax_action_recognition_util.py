import ailia
import math
import numpy as np

TIME_RANGE = 15

def pose_postprocess(pose_keypoints):
    thre = 0.2
    pose_keypoints[:, :, 0:2] = pose_keypoints[:, :, 0:2] - 0.5
    # pose_keypoints[:, :, 0][pose_keypoints[:, :, 2] == 0] = 0
    # pose_keypoints[:, :, 1][pose_keypoints[:, :, 2] == 0] = 0
    pose_keypoints[:, :, 0][pose_keypoints[:, :, 2] < thre] = 0
    pose_keypoints[:, :, 1][pose_keypoints[:, :, 2] < thre] = 0
    pose_keypoints[:, :, 2][pose_keypoints[:, :, 2] < thre] = 0
    return pose_keypoints