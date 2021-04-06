import sys

import time
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
IMAGE_PATH = 'images/000002_LR.png'
SAVE_IMAGE_PATH = 'images/output.png'
IMAGE_HEIGHT = 194    # net.get_input_shape()[3]
IMAGE_WIDTH = 194     # net.get_input_shape()[2]

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Single Image Super-Resolution with HAN', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-n', '--normal', action='store_true',
    help=('By default, the optimized model is used, but with this option, ' +
          'you can switch to the normal (not optimized) model.')
)
parser.add_argument(
    '--scale', default=2, type=int, choices=[2, 3, 4, 8],
    help=('Super-resolution scale. By default 2 (generates an image with twice the resolution).')
)
parser.add_argument(
    '--blur', action='store_true',
    help=('By default, uses the model trained on images degraded with the Bicubic (BI) Degradation Model, ' + 
          'but with this option, you can switch to the model trained on images degraded with the Blur-downscale Degradation Model (BD). ' +
          'A scale of 3 can only be used in combination with this option.')
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if args.blur:
    args.scale = 3
    if not args.normal:
        WEIGHT_PATH = 'han_BDX3.opt.onnx'
        MODEL_PATH = 'han_BDX3.opt.onnx.prototxt'
    else:
        WEIGHT_PATH = 'han_BDX3.onnx'
        MODEL_PATH = 'han_BDX3.onnx.prototxt'
else:
    if args.scale == 2:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX2.opt.onnx'
            MODEL_PATH = 'han_BIX2.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX2.onnx'
            MODEL_PATH = 'han_BIX2.onnx.prototxt'
    elif args.scale == 3:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX3.opt.onnx'
            MODEL_PATH = 'han_BIX3.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX3.onnx'
            MODEL_PATH = 'han_BIX3.onnx.prototxt'
    elif args.scale == 4:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX4.opt.onnx'
            MODEL_PATH = 'han_BIX4.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX4.onnx'
            MODEL_PATH = 'han_BIX4.onnx.prototxt'
    elif args.scale == 8:
        if not args.normal:
            WEIGHT_PATH = 'han_BIX8.opt.onnx'
            MODEL_PATH = 'han_BIX8.opt.onnx.prototxt'
        else:
            WEIGHT_PATH = 'han_BIX8.onnx'
            MODEL_PATH = 'han_BIX8.onnx.prototxt'
    else:
        logger.info('Incorrect scale (choose from 2, 3, 4 or 8).')
        exit(-1)

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/han/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    logger.info('Model: ' + WEIGHT_PATH[:-5])
    logger.info('Scale: ' + str(args.scale))
    
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info('Input image: ' + image_path)

        img = cv2.imread(image_path)
        IMAGE_HEIGHT = img.shape[0]
        IMAGE_WIDTH = img.shape[1]

        net.set_input_shape((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

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
