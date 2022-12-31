import sys
import time
import struct

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
# logger
from logging import getLogger  # noqa

from v2v_util import voxelize, evaluate_keypoints

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "msra-subject3-epoch15.onnx"
MODEL_PATH = "msra-subject3-epoch15.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/v2v-posenet/'

DEFAULT_DEPTH = 'cvpr15_MSRAHandGestureDB/P3/1/000000_depth.bin'

IMG_WIDTH = 320
IMG_HEIGHT = 240
MAX_DEPTH = 700

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'V2V-PoseNet', None, None
)
parser.add_argument(
    '--input', '-i', default=DEFAULT_DEPTH
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================

def load_depthmap(filename):
    with open(filename, mode='rb') as f:
        data = f.read()
        _, _, left, top, right, bottom = struct.unpack('I' * 6, data[:6 * 4])
        num_pixel = (right - left) * (bottom - top)
        cropped_image = struct.unpack('f' * num_pixel, data[6 * 4:])

        cropped_image = np.asarray(cropped_image).reshape(bottom - top, -1)
        depth_image = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        depth_image[top:bottom, left:right] = cropped_image
        depth_image[depth_image == 0] = MAX_DEPTH

        return depth_image


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy)
    return points


# ======================
# Main functions
# ======================

def predict(net, points, refpoint):
    input = voxelize(points, refpoint)
    inputs = np.expand_dims(input, axis=0)
    inputs = inputs.astype(np.float32)
    refpoint = np.array([refpoint])

    # feedforward
    if not args.onnx:
        output = net.predict([inputs])
    else:
        output = net.run(None, {
            'inputs': inputs
        })

    heatmaps = output[0]
    keypoints = evaluate_keypoints(heatmaps, refpoint)

    return keypoints


def recognize_from_points(net):
    # input image loop
    for file_path in args.input:
        logger.info(file_path)

        # prepare input data
        depthmap = load_depthmap(file_path)
        fx = fy = 241.42
        points = depthmap2points(depthmap, fx, fy)
        points = points.reshape((-1, 3))
        refpoint = [-22.2752, -66.7133, 320.7968]

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pred = predict(net, points, refpoint)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            pred = predict(net, points, refpoint)

        # res_img = draw_keypoints(img, pred)
        #
        # # plot result
        # savepath = get_savepath(args.savepath, file_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    recognize_from_points(net)


if __name__ == '__main__':
    main()
