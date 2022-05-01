import sys
import time

import numpy as np
import cv2
from PIL import Image

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


def predict(net_det, net_pred, net_ref, img, img_info):
    img, img_shape, scale_factor = preprocess(img)
    img_shape = np.array(img_shape, dtype=np.int64)
    scale_factor = np.array(scale_factor, dtype=np.float32)

    # feedforward
    if not args.onnx:
        output = net_det.predict([img, img_shape, scale_factor])
    else:
        output = net_det.run(None, {'img': img, 'img_shape': img_shape, 'scale_factor': scale_factor})

    det_bboxes, det_labels, embeds, det_depths, det_depths_uncertainty, det_dims, det_alphas, det_2dcs = output
    print(det_bboxes)
    print(det_bboxes.shape)
    print(img_info)

    # pred = post_processing(output)
    #
    # return pred


def recognize_from_image(net_det, net_pred, net_ref):
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
                out = predict(net_det, net_pred, net_ref, img, img_info)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net_det, net_pred, net_ref, img, img_info)

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
        net_pred = ailia.Net(MODEL_MOTION_PRED_PATH, WEIGHT_MOTION_PRED_PATH, env_id=env_id)
        net_ref = ailia.Net(MODEL_MOTION_RFINE_PATH, WEIGHT_MOTION_RFINE_PATH, env_id=env_id)
    else:
        import onnxruntime
        net_det = onnxruntime.InferenceSession(WEIGHT_DETECTOR_PATH)
        net_pred = onnxruntime.InferenceSession(WEIGHT_MOTION_PRED_PATH)
        net_ref = onnxruntime.InferenceSession(WEIGHT_MOTION_RFINE_PATH)

    recognize_from_image(net_det, net_pred, net_ref)


if __name__ == '__main__':
    main()
