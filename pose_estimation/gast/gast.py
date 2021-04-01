import os
import sys
import time
import json

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from gast_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_27FRAME_17JOINT_PATH = '27_frame_17_joint_model.onnx'
MODEL_27FRAME_17JOINT_PATH = '27_frame_17_joint_model.onnx.prototxt'
WEIGHT_27FRAME_19JOINT_PATH = '27_frame_19_joint_model.onnx'
MODEL_27FRAME_19JOINT_PATH = '27_frame_19_joint_model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gast/'

VIDEO_PATH = './data/baseball.mp4'
SAVE_PATH = 'output.mp4'

ROT = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('GAST model', VIDEO_PATH, SAVE_PATH)
parser.add_argument(
    '-k', '--keypoints_file', default='./data/baseball.json', metavar='NAME',
    help='The path of keypoints file'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def load_json(file_path, num_joints, num_person=2):
    with open(file_path, 'r') as fr:
        video_info = json.load(fr)

    # Loading whole-body keypoints including body(17)+hand(42)+foot(6)+facial(68) joints
    # 2D Whole-body human pose estimation paper: https://arxiv.org/abs/2007.11858
    if num_joints == 19:
        num_joints_revise = 133
    else:
        num_joints_revise = 17

    label = video_info['label']
    label_index = video_info['label_index']

    num_frames = video_info['data'][-1]['frame_index']
    keypoints = np.zeros((num_person, num_frames, num_joints_revise, 2), dtype=np.float32)
    scores = np.zeros((num_person, num_frames, num_joints_revise), dtype=np.float32)

    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']

        for index, skeleton_info in enumerate(frame_info['skeleton']):
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            bbox = skeleton_info['bbox']

            if len(bbox) == 0 or index + 1 > num_person:
                continue

            pose = np.asarray(pose, dtype=np.float32)
            score = np.asarray(score, dtype=np.float32)
            score = score.reshape(-1)

            keypoints[index, frame_index - 1] = pose
            scores[index, frame_index - 1] = score

    if num_joints != num_joints_revise:
        # body(17) + foot(6) = 23
        return keypoints[:, :, :23], scores[:, :, :23], label, label_index
    else:
        return keypoints, scores, label, label_index


def recognize_from_video(net, params):
    video_file = args.input[0]
    cap = get_capture(video_file)
    keypoints_file = args.keypoints_file

    num_joints = params["num_joints"]

    logger.info('Loading 2D keypoints ...')
    keypoints, scores, _, _ = load_json(keypoints_file, num_joints)
    keypoints = keypoints[0]

    keypoints, valid_frames = coco_h36m(keypoints)

    # Get the width and height of video
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # normalize keypoints
    input_keypoints = normalize_screen_coordinates(
        keypoints[..., :2], w=width, h=height)
    # print("input_keypoints---", input_keypoints)

    filter_widths = [3, 3, 3]
    channels = 128
    # print("filter_widths---", filter_widths)
    # print("channels---", channels)

    x = receptive_field(filter_widths)
    pad = (x - 1) // 2  # Padding on each side
    causal_shift = 0

    joints_left, joints_right, h36m_skeleton, keypoints_metadata = get_joints_info(num_joints)
    kps_left, kps_right = joints_left, joints_right

    generator = DataLoader(
        [input_keypoints[valid_frames]],
        pad=pad, causal_shift=causal_shift,
        kps_left=kps_left, kps_right=kps_right
    )
    for batch_2d in generator.next_epoch():
        in_name = net.get_inputs()[0].name
        out_name = net.get_outputs()[0].name
        output = net.run([out_name],
                         {in_name: batch_2d})
        predicted_3d_pos = output[0]

        predicted_3d_pos[1, :, :, 0] *= -1
        predicted_3d_pos[1, :, joints_left + joints_right] = \
            predicted_3d_pos[1, :, joints_right + joints_left]
        predicted_3d_pos = np.mean(predicted_3d_pos, axis=0, keepdims=True)
        predicted_3d_pos = predicted_3d_pos.squeeze(0)
        break

    prediction = camera_to_world(predicted_3d_pos, R=ROT, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    prediction_new = np.zeros((*input_keypoints.shape[:-1], 3), dtype=np.float32)
    prediction_new[valid_frames] = prediction

    logger.info('Rendering ...')
    anim_output = {'Reconstruction': prediction_new}
    render_animation(
        keypoints, keypoints_metadata, anim_output, h36m_skeleton, 25, 3000,
        np.array(70., dtype=np.float32), args.savepath,
        cap, viewport=(width, height),
        downsample=1, size=5)

    logger.info('Script finished successfully.')


def main():
    info = {
        17: (
            WEIGHT_27FRAME_17JOINT_PATH, MODEL_27FRAME_17JOINT_PATH),
        19: (
            WEIGHT_27FRAME_19JOINT_PATH, MODEL_27FRAME_19JOINT_PATH),
    }
    # model files check and download
    # weight_path, model_path = info[0]
    weight_path, model_path = info[17]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    num_joints = 17
    params = {
        "num_joints": num_joints,
    }

    # net initialize
    import onnxruntime
    net = onnxruntime.InferenceSession(weight_path)

    recognize_from_video(net, params)


if __name__ == '__main__':
    main()
