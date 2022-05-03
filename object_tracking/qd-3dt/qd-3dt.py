import sys
import time

import numpy as np
import cv2
from pyquaternion import Quaternion

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from video_utils import load_annotations
import tracking_utils as tu
from motion_tracker import MotionTracker
import tracker_model

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DETECTOR_PATH = 'nuScenes_3Dtracking.onnx'
MODEL_DETECTOR_PATH = 'nuScenes_3Dtracking.onnx.prototxt'
WEIGHT_MOTION_PRED_PATH = 'nuScenes_LSTM_motion_pred.onnx'
MODEL_MOTION_PRED_PATH = 'nuScenes_LSTM_motion_pred.onnx.prototxt'
WEIGHT_MOTION_RFINE_PATH = 'nuScenes_LSTM_motion_refine.onnx'
MODEL_MOTION_RFINE_PATH = 'nuScenes_LSTM_motion_refine.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/qd-3dt/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_MAX = 1600
IMAGE_MIN = 900

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Monocular Quasi-Dense 3D Object Tracking', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = img.shape[:2]
    divisor = 32

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # rescale
    scale_factor = min(IMAGE_MAX / max(h, w), IMAGE_MIN / min(h, w))
    ow, oh = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
    if ow != w or oh != h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img_shape = img.shape

    mean = np.array([106.07, 127.705, 128.08])
    std = np.array([73.458, 70.129, 70.463])
    img = (img - mean) / std

    pad_h = int(np.ceil(oh / divisor)) * divisor
    pad_w = int(np.ceil(ow / divisor)) * divisor
    if pad_w != ow or pad_h != oh:
        pad_img = np.zeros((pad_h, pad_w, 3), dtype=img.dtype)
        pad_img[:oh, :ow, ...] = img
        img = pad_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, img_shape, scale_factor


def post_processing(output):
    return None


def predict(net_det, lstm_pred, lstm_refine, img, img_info):
    img, img_shape, scale_factor = preprocess(img)

    track_config = {
        'lstm_pred': lstm_pred,
        'lstm_refine': lstm_refine,
        'init_score_thr': 0.8,
        'init_track_id': 0,
        'obj_score_thr': 0.5,
        'match_score_thr': 0.5,
        'memo_tracklet_frames': 10,
        'memo_backdrop_frames': 1,
        'memo_momentum': 0.8,
        'motion_momentum': 0.9,
        'nms_conf_thr': 0.5,
        'nms_backdrop_iou_thr': 0.3,
        'nms_class_iou_thr': 0.7,
        'loc_dim': 7,
        'with_deep_feat': True,
        'with_cats': True,
        'with_bbox_iou': True,
        'with_depth_ordering': True,
        'track_bbox_iou': 'box3d',
        'depth_match_metric': 'motion',
        'match_metric': 'cycle_softmax',
        'match_algo': 'greedy',
        'with_depth_uncertainty': True
    }
    if predict.tracker is None:
        predict.tracker = MotionTracker(**track_config)
    elif img_info.get('first_frame', False):
        num_tracklets = predict.tracker.num_tracklets
        track_config['init_track_id'] = num_tracklets
        del predict.tracker
        predict.tracker = MotionTracker(**track_config)

    calib = img_info['cali']
    ori_shape = (img_info['height'], img_info['width'], 3)
    if ori_shape != img_shape:
        focal_length = calib[0][0]
        width = img_shape[1]
        height = img_shape[0]
        calib = [[focal_length * scale_factor, 0, width / 2.0, 0],
                 [0, focal_length * scale_factor, height / 2.0, 0],
                 [0, 0, 1, 0]]

    img_info['calib'] = calib
    img_shape = np.array(img_shape, dtype=np.int64)
    scale_factor = np.array(scale_factor, dtype=np.float32)

    # feedforward
    if not args.onnx:
        output = net_det.predict([img, img_shape, scale_factor])
    else:
        output = net_det.run(None, {'img': img, 'img_shape': img_shape, 'scale_factor': scale_factor})

    det_bboxes, det_labels, embeds, det_depths, det_depths_uncertainty, det_dims, det_alphas, det_2dcs = output

    # TODO: use boxes_3d to match KF3d in tracker
    projection = np.array(img_info['calib'])
    position = np.array(img_info['pose']['position'])
    r_camera_to_world = tu.angle2rot(np.array(img_info['pose']['rotation']))
    rotation = np.array(r_camera_to_world)
    cam_rot_quat = Quaternion(matrix=r_camera_to_world)

    corners = tu.imagetocamera(det_2dcs, det_depths, projection)
    corners_global = tu.cameratoworld(corners, position, rotation)
    det_yaws = tu.alpha2yaw(
        det_alphas, corners[:, 0:1], corners[:, 2:3])

    quat_det_yaws_world = {'roll_pitch': [], 'yaw_world': []}
    for det_yaw in det_yaws:
        yaw_quat = Quaternion(
            axis=[0, 1, 0], radians=det_yaw)
        rotation_world = cam_rot_quat * yaw_quat
        if rotation_world.z < 0:
            rotation_world *= -1
        roll_world, pitch_world, yaw_world = tu.quaternion_to_euler(
            rotation_world.w, rotation_world.x, rotation_world.y,
            rotation_world.z)
        quat_det_yaws_world['roll_pitch'].append(
            [roll_world, pitch_world])
        quat_det_yaws_world['yaw_world'].append(yaw_world)

    det_yaws_world = np.array(quat_det_yaws_world['yaw_world'])[:, None]
    det_boxes_3d = np.concatenate([
        corners_global, det_yaws_world, det_dims
    ], axis=1)

    frame_ind = img_info.get('index', -1)
    pure_det = False
    match_bboxes, match_labels, match_boxes_3ds, ids, inds, valids = \
        predict.tracker.match(
            bboxes=det_bboxes,
            labels=det_labels,
            boxes_3d=det_boxes_3d,
            depth_uncertainty=det_depths_uncertainty,
            position=position,
            rotation=rotation,
            embeds=embeds,
            cur_frame=frame_ind,
            pure_det=pure_det)

    # pred = post_processing(output)
    #
    # return pred


predict.tracker = None


def recognize_from_image(net_det, lstm_pred, lstm_ref):
    img_infos = load_annotations('tracking_val.json')

    img_infos = [x for x in img_infos if x['video_id'] == 0]

    # input image loop
    for img_info in img_infos:
        image_path = img_info['filename']
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(net_det, lstm_pred, lstm_ref, img, img_info)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net_det, lstm_pred, lstm_ref, img, img_info)

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking DETECTOR model...')
    check_and_download_models(WEIGHT_DETECTOR_PATH, MODEL_DETECTOR_PATH, REMOTE_PATH)
    logger.info('Checking MOTION_PRED model...')
    check_and_download_models(WEIGHT_MOTION_PRED_PATH, MODEL_MOTION_PRED_PATH, REMOTE_PATH)
    logger.info('Checking MOTION_REFINE model...')
    check_and_download_models(WEIGHT_MOTION_RFINE_PATH, MODEL_MOTION_RFINE_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_det = ailia.Net(MODEL_DETECTOR_PATH, WEIGHT_DETECTOR_PATH, env_id=env_id)
        lstm_pred = ailia.Net(MODEL_MOTION_PRED_PATH, WEIGHT_MOTION_PRED_PATH, env_id=env_id)
        lstm_ref = ailia.Net(MODEL_MOTION_RFINE_PATH, WEIGHT_MOTION_RFINE_PATH, env_id=env_id)
    else:
        import onnxruntime
        net_det = onnxruntime.InferenceSession(WEIGHT_DETECTOR_PATH)
        lstm_pred = onnxruntime.InferenceSession(WEIGHT_MOTION_PRED_PATH)
        lstm_ref = onnxruntime.InferenceSession(WEIGHT_MOTION_RFINE_PATH)
        tracker_model.onnx = True

    recognize_from_image(net_det, lstm_pred, lstm_ref)


if __name__ == '__main__':
    main()
