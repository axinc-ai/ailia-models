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

WEIGHT_PATH = 'vehicle-attributes-recognition-barrier-0042.onnx'
MODEL_PATH = 'vehicle-attributes-recognition-barrier-0042.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vehicle-attributes-recognition-barrier/'

IMAGE_PATH = 'demo.png'
IMAGE_SIZE = 72

COLOR_LIST = (
    'white', 'gray', 'yellow', 'red', 'green', 'blue', 'black'
)

TYPE_LIST = (
    'car', 'van', 'truck', 'bus'
)

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'vehicle-attributes-recognition-barrier', IMAGE_PATH, None,
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

        out_typ, out_clr = output
        typ = TYPE_LIST[np.argmax(out_typ)]
        clr = COLOR_LIST[np.argmax(out_clr)]

        logger.info("- Type: %s" % typ)
        logger.info("- Color: %s" % clr)

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

    recognize_from_image(net)


if __name__ == '__main__':
    main()
