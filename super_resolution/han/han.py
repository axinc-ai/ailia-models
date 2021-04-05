import sys

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = '000002_LR.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 194    # net.get_input_shape()[3]
IMAGE_WIDTH = 194     # net.get_input_shape()[2]
OUTPUT_HEIGHT = 194*2  # net.get_output_shape()[3]
OUTPUT_WIDTH = 194*2   # net.get_output.shape()[2]


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Single Image Super-Resolution with HAN', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal', action='store_true',
    help=('By default, the optimized model is used, but with this option, ' +
          'you can switch to the normal (not optimized) model')
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
"""
if not args.normal:
    WEIGHT_PATH = 'han_BIX2.onnx.opt.onnx'
    MODEL_PATH = 'han_BIX2.opt.onnx.prototxt'
else:
    WEIGHT_PATH = 'han_BIX2.onnx'
    MODEL_PATH = 'han_BIX2.onnx.prototxt'
"""

WEIGHT_PATH = 'han_BIX2.onnx'
MODEL_PATH = 'han_BIX2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/han/'

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    net.set_input_shape((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='255',
            gen_input_ailia=True,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # postprocessing
        output_img = preds_ailia[0].transpose((1, 2, 0))
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img * 255)
        
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    recognize_from_image()


if __name__ == '__main__':
    main()
