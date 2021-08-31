import sys
import time

import cv2
import numpy as np

import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'age-gender-recognition-retail-0013.onnx'
MODEL_PATH = 'age-gender-recognition-retail-0013.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/age-gender-recognition/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 62

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'age-gender-recognition', IMAGE_PATH, SAVE_IMAGE_PATH,
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def recognize_from_image(net):
    # prepare input data
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(
            image_path, (IMAGE_SIZE, IMAGE_SIZE),
            normalize_type='None')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = img / 255
        img = np.expand_dims(img, axis=0)  # 次元合せ

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict([img])
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict([img])

        prob, age_conv3 = output
        prob = prob[0][0][0]
        age_conv3 = age_conv3[0][0][0][0]

        i = np.argmax(prob)
        logger.info(" gender is: %s (%.2f)" % ('Female' if i == 0 else 'Male', prob[i] * 100))
        logger.info(" age is: %d" % round(age_conv3 * 100))

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )

    # load model
    env_id = ailia.get_gpu_environment_id()
    logger.info(f'env_id: {env_id}')

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )

    # image mode
    recognize_from_image(net)


if __name__ == '__main__':
    main()
