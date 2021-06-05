import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from va_cnn_utils import *
from labels import LABELS

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'va-cnn.onnx'
MODEL_PATH = 'va-cnn.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/va-cnn/'

INPUT_FILE = 'data/ntu/nturgb+d_skeletons/S001C001P001R001A001.skeleton'
SAVE_IMAGE_PATH = None

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('View Adaptive Neural Networks', INPUT_FILE, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(x):
    max = 5.18858098984
    min = -5.28981208801

    imgs, maxmin = torgb([x], max, min)
    imgs = np.stack([imgs[i] for i in range(len(imgs))], axis=0)
    maxmin = np.vstack(maxmin).astype(np.float32)

    return imgs, maxmin


def recognize_from_keypoints(net):
    import h5py
    f = h5py.File("NTU_CS.h5", 'r')
    test_X = f['test_x'][:]
    x = test_X[0]

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare grand truth
        imgs, maxmin = preprocess(x)

        logger.debug(f'input image shape: {imgs.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict({'imgs': imgs, 'maxmin': maxmin})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict({'imgs': imgs, 'maxmin': maxmin})

        pred, img, trans = output

        i = np.argmax(pred, axis=-1)[0]
        label = LABELS[i]
        logger.info('Action estimate -> ' + label)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    recognize_from_keypoints(net)


if __name__ == '__main__':
    main()
