import numpy as np
import time
import os
import sys
import cv2
from PIL import Image

from mlsd_utils import pred_lines

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from image_utils import load_image
import webcamera_utils

# logger
from logging import getLogger

logger = getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mlsd/'
WEIGHT_PATH = 'M-LSD_512_large.opt.onnx'
MODEL_PATH = 'M-LSD_512_large.opt.onnx.prototxt'
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('mlsd model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)

# ======================
# Main functions
# ======================

def gradio_wrapper_for_LSD(img_input, net):
  lines = pred_lines(img_input, net, input_shape=[512, 512])
  img_output = img_input.copy()

  # draw lines
  for line in lines:
    x_start, y_start, x_end, y_end = [int(val) for val in line]
    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [0,255,255], 2)

  return img_output

def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    net.set_input_shape((1, IMAGE_WIDTH, IMAGE_HEIGHT, 4))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='ImageNet',
            gen_input_ailia=True,
        )
        img_input = input_data[0]
        img_input = np.transpose(img_input, (1, 2, 0)) * 255
        img_input = np.array(Image.open(image_path))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds_img = gradio_wrapper_for_LSD(img_input, net)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_img = gradio_wrapper_for_LSD(img_input, net)

        # postprocessing
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, preds_img)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='ImageNet'
        )
        img_input = img_input[0]
        img_input = np.transpose(img_input, (1, 2, 0)) * 255

        # inference
        preds_img = gradio_wrapper_for_LSD(img_input, net)
        cv2.imshow('frame', preds_img)

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
