import os
import sys
import time
import json

import numpy as np
import cv2
from tqdm import tqdm

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

WEIGHT_YOLOV3_PATH = 'yolov3.opt.onnx'
MODEL_YOLOV3_PATH = 'yolov3.opt.onnx.prototxt'
REMOTE_YOLOV3_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'
WEIGHT_POSE_PATH = 'pose_hrnet_w48_384x288.onnx'
MODEL_POSE_PATH = 'pose_hrnet_w48_384x288.onnx.prototxt'
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


def gen_kpts(frames, yolo_model, pose_model, num_peroson=1):
    # collect keypoints coordinate
    logger.info('Generating 2D pose ...')

    people_sort = ObjSort()

    kpts_result = []
    scores_result = []
    for i in tqdm(range(len(frames))):
        frame = frames[i]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bboxs, scores = yolo_human_det(img, yolo_model)
        if not bboxs.any():
            continue

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        inputs, origin_img, center, scale = preprocess(frame, track_bboxs, num_peroson)
        inputs = inputs[:, [2, 1, 0]]

        output = pose_model.predict({'input': inputs})[0]

        # compute coordinate
        preds, maxvals = get_final_preds(output, np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores


def yolo_human_det(img, detector, confidence=0.70, nms_thresh=0.4):
    detector.compute(img, confidence, nms_thresh)

    h, w = img.shape[:2]
    bboxs = []
    scores = []

    count = detector.get_object_count()
    for idx in range(count):
        obj = detector.get_object(idx)
        if obj.category != 0:
            continue

        bboxs.append(np.array([
            obj.x * w,
            obj.y * h,
            (obj.x + obj.w) * w,
            (obj.x + obj.h) * h,
        ]))
        scores.append(obj.prob)

    bboxs = np.asarray(bboxs)
    scores = np.array(scores).reshape(-1, 1)

    return bboxs, scores


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : (x1, y1, x2, y2)
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)
    x1, y1, x2, y2 = box[:4]
    box_width, box_height = x2 - x1, y2 - y1

    center[0] = x1 + box_width * 0.5
    center[1] = y1 + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def preprocess(img, bboxs, num_pos=2):
    IMAGE_SIZE = (288, 384)

    inputs = []
    centers = []
    scales = []
    for bbox in bboxs[:num_pos]:
        c, s = box_to_center_scale(bbox, img.shape[0], img.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, IMAGE_SIZE)
        input = cv2.warpAffine(
            img,
            trans,
            (IMAGE_SIZE[0], IMAGE_SIZE[1]),
            flags=cv2.INTER_LINEAR)

        input = normalize_image(input.astype(np.float32), 'ImageNet')
        input = input.transpose(2, 0, 1)  # HWC -> CHW
        input = np.expand_dims(input, axis=0)
        inputs.append(input)

    inputs = np.vstack(inputs)
    return inputs, img, centers, scales


def recognize_from_video(net, info):
    keypoints_file = args.keypoints_file

    video_file = args.input[0]
    cap = get_capture(video_file)
    assert cap.isOpened(), 'Cannot capture source'

    # Get the width and height of video
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Load video frame
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(video_length):
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)

    num_person = info["num_person"]
    keypoints, scores = gen_kpts(
        frames, info["yolo_model"], info["pose_net"],
        num_peroson=num_person)
    # keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # re_kpts = revise_kpts(keypoints, scores, valid_frames)
    # num_person = len(re_kpts)

    num_joints = info["num_joints"]

    logger.info('Loading 2D keypoints ...')
    keypoints, scores, _, _ = load_json(keypoints_file, num_joints)
    keypoints = keypoints[0]

    keypoints, valid_frames = coco_h36m(keypoints)

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
    # model files check and download
    logger.info("=== YOLOv3 model ===")
    check_and_download_models(WEIGHT_YOLOV3_PATH, MODEL_YOLOV3_PATH, REMOTE_YOLOV3_PATH)
    logger.info("=== HRNet model ===")
    check_and_download_models(WEIGHT_POSE_PATH, MODEL_POSE_PATH, REMOTE_PATH)

    model_info = {
        17: (
            WEIGHT_27FRAME_17JOINT_PATH, MODEL_27FRAME_17JOINT_PATH),
        19: (
            WEIGHT_27FRAME_19JOINT_PATH, MODEL_27FRAME_19JOINT_PATH),
    }
    logger.info("=== GAST model ===")
    weight_path, model_path = model_info[17]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    num_joints = 17
    num_person = 2

    # net initialize
    import onnxruntime
    net = onnxruntime.InferenceSession(weight_path)

    detector = ailia.Detector(
        MODEL_YOLOV3_PATH,
        WEIGHT_YOLOV3_PATH,
        80,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id,
    )
    pose_net = ailia.Net(MODEL_POSE_PATH, WEIGHT_POSE_PATH, env_id=args.env_id)

    info = {
        "yolo_model": detector,
        "pose_net": pose_net,
        "num_joints": num_joints,
        "num_person": num_person,
    }

    recognize_from_video(net, info)


if __name__ == '__main__':
    main()
