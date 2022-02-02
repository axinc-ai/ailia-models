import sys
import time

import numpy as np
import cv2

import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_writer, get_capture  # noqa: E402

from real_esrgan_utils import RealESRGAN

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
INPUT_IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

W = 256
H = 256

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/real-esrgan/'

# =======================
# Arguments Parser Config
# =======================
parser = get_base_parser(
    'Real-ESRGAN',
    INPUT_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', metavar='MODEL_NAME',
    default='RealESRGAN',
    help='[RealESRGAN, RealESRGAN_anime]'
)


args = update_parser(parser)

MODEL_PATH = args.model + '.opt.onnx.prototxt'
WEIGHT_PATH = args.model + '.opt.onnx'


def enhance_image():
    # prepare input data
    img = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dsize=(H, W))

    # net initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    upsampler = RealESRGAN(model)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = upsampler.enhance(img)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        output = upsampler.enhance(img)

    cv2.imwrite(args.savepath, output)

    logger.info('Script finished successfully.')


def enhance_video():
    # net initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    upsampler = RealESRGAN(model)

    capture = get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = calc_adjust_fsize(f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH)
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = cv2.resize(frame, dsize=(H, W))

        # inference
        output = upsampler.enhance(img)

        #plot result
        cv2.imshow('frame', output)

        if writer is not None:
            writer.release()
        logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video:
        enhance_video()
    else:
        enhance_image()


if __name__ == '__main__':
    main()
