import sys
import os
import time
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

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

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'd4lcn.onnx'
MODEL_PATH = 'd4lcn.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/d4lcn/'

IMAGE_PATH = '000001.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'D4LCN', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--depth_path', type=str, default=None,
    help='the label file (object labels for image) or stored directory path.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_depth(path, name):
    if path is None:
        file_path = "depth/%s.png" % name
        if os.path.exists(file_path):
            logger.info("depth file: %s" % file_path)
            return file_path
        else:
            return None
    elif os.path.isdir(path):
        file_path = "%s/%s.png" % (path, name)
        if os.path.exists(file_path):
            logger.info("depth file: %s" % file_path)
            return file_path
    elif os.path.exists(path):
        logger.info("depth file: %s" % path)
        return path

    logger.error("depth file is not found. (path: %s)" % path)
    sys.exit(-1)


# ======================
# Main functions
# ======================


def preprocess(img, depth):
    h, w = img.shape[:2]

    scale_factor = IMAGE_HEIGHT / h
    h = np.round(h * scale_factor).astype(int)
    w = np.round(w * scale_factor).astype(int)

    # resize
    img = cv2.resize(img.astype(np.float32), (w, h))
    depth = cv2.resize(depth.astype(np.float32), (w, h))

    # Normalize
    mean = np.array([102.9801, 115.9465, 122.7717])
    std = np.array([1., 1., 1.])
    depth_mean = np.array([4413.1606, 4413.1606, 4413.1606])
    depth_std = np.array([3270.0159, 3270.0159, 3270.0159])
    img = (img - mean) / std
    depth = (depth - depth_mean) / depth_std

    img = np.transpose(img, [2, 0, 1])
    depth = np.transpose(depth, [2, 0, 1])

    img = np.expand_dims(img, axis=0)
    depth = np.expand_dims(depth, axis=0)

    return img, depth


def predict(net, img, depth):
    h, w = img.shape[:2]
    img, depth = preprocess(img, depth)

    output = net.predict([img, depth])

    cls, prob, bbox_2d, bbox_3d = output

    return


def recognize_from_image(net):
    depth_path = args.depth_path

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # depth
        name = os.path.splitext(os.path.basename(image_path))[0]
        depth_path = get_depth(depth_path, name)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth[:, :, np.newaxis]
        depth = np.tile(depth, (1, 1, 3))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(net, img, depth)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net, img, depth)

        save_path = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {save_path}')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
