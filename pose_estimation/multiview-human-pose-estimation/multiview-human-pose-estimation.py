import os
import sys
import time
import numpy as np
import cv2

from lib_mhpe.vis import show_heatmaps

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer, \
    calc_adjust_fsize, preprocess_frame  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'multiview-human-pose-estimation.opt.onnx'
MODEL_PATH = 'multiview-human-pose-estimation.opt.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/multiview-human-pose-estimation/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('multiview-human-pose-estimation model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        raw_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.resize(raw_img, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255
        img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        raw_img = raw_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        logger.info(f'input image shape: {img.shape}')

        net.set_input_shape(img.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                result = net.run(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result = net.run(img)

        heatmaps = result[0]

        savepath = get_savepath(args.savepath, image_path, ext='.jpg')
        logger.info(f'saved at : {savepath}')
        grid_image, _, _ = show_heatmaps(img, raw_img, heatmaps, normalize=True)

        cv2.imwrite(savepath, grid_image)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        s = max(f_h, f_w)
        writer = get_writer(args.savepath, s, s)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = frame
        raw_img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255
        img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        net.set_input_shape(img.shape)
        result = net.run(img)
        heatmaps = result[0]

        grid_image, _, pose_raw_image = show_heatmaps(img, raw_img, heatmaps, normalize=True)
        #cv2.imshow('Plot Pose 2D', grid_image)
        cv2.imshow('Plot Pose 2D', pose_raw_image)

        if writer is not None:
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
