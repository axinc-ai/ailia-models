import time
import os
import sys
import cv2
import glob
import numpy as np
from scipy.special import softmax

from roneld_utils import roneld_lane_detection, ScaleNew, Normalize

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
import webcamera_utils

# logger
from logging import getLogger

logger = getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/erfnet/'
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

RESIZE_MODE_LISTS = ['padding', 'crop']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('roneld model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-r', '--resize', metavar='RESIZE',
    default='crop', choices=RESIZE_MODE_LISTS,
    help='resize mode lists: ' + ' | '.join(RESIZE_MODE_LISTS)
)
args = update_parser(parser)

WEIGHT_PATH = 'erfnet.opt.onnx'
MODEL_PATH = 'erfnet.opt.onnx.prototxt'

HEIGHT = 208
WIDTH = 976

INPUT_MEAN = [103.939, 116.779, 123.68]
INPUT_STD = [1, 1, 1]

# ======================
# Main functions
# ======================

def crop_and_resize(raw_img):
    if args.resize=="padding":
        #add padding
        frame,resized_img = webcamera_utils.adjust_frame_size(raw_img, HEIGHT, WIDTH)
        return resized_img
    elif args.resize=="crop":
        #cut top
        scale_x = (WIDTH / raw_img.shape[1])
        crop_y = raw_img.shape[0] * scale_x - HEIGHT
        crop_y = int(crop_y / scale_x)

        img = raw_img[crop_y:, :, :]  #keep aspect
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_LINEAR)
        return img
    return None

def recognize_from_image():
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 3, HEIGHT, WIDTH))

    prev_lanes = []
    prev_curves = np.zeros(10)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        trans = Normalize(mean=(INPUT_MEAN, (0,)), std=(INPUT_STD, (1,)))

        raw_img = crop_and_resize(raw_img)
        img = raw_img

        #img = cv2.resize(raw_img, (WIDTH, HEIGHT))

        img = np.expand_dims(img, 0)
        img = trans(img)
        img = np.array(img).transpose(0, 3, 1, 2)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                output, output_exist = net.run(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            output, output_exist = net.run(img)

        output = softmax(output, axis=1)
        lane_images = []

        for num in range(4):
            lane_image = output[0][num + 1]
            lane_image = (lane_image * 255).astype(int)
            lane_images.append(lane_image)

        # call to roneld and store output for next method call
        output_images, prev_lanes, prev_curves, curve_mode = \
            roneld_lane_detection(lane_images, prev_lanes, prev_curves, curve_mode=False,
                                  image=raw_img)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, raw_img)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = webcamera_utils.get_capture(args.video)

    prev_lanes = []
    prev_curves = np.zeros(10)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        trans = Normalize(mean=(INPUT_MEAN, (0,)), std=(INPUT_STD, (1,)))

        frame = crop_and_resize(frame)
        img = frame

        #img = cv2.resize(frame, (WIDTH, HEIGHT))
        img = np.expand_dims(img, 0)
        img = trans(img)
        img = np.array(img).transpose(0, 3, 1, 2)

        output, output_exist = net.run(img)
        output = softmax(output, axis=1)
        lane_images = []

        for num in range(4):
            lane_image = output[0][num + 1]
            lane_image = (lane_image * 255).astype(int)
            lane_images.append(lane_image)

        # call to roneld and store output for next method call
        output_images, prev_lanes, prev_curves, curve_mode = \
            roneld_lane_detection(lane_images, prev_lanes, prev_curves, curve_mode=False,
                                  image=frame)

        cv2.imshow('frame', frame)

        # save results
        if writer is not None:
            writer.write(frame)

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