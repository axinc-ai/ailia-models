import numpy as np
import time
import os
import sys
import cv2
from scipy.special import softmax

from erfnet_utils import ScaleNew, Normalize

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

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('erfnet model', IMAGE_PATH, SAVE_IMAGE_PATH)
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
def recognize_from_image():
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 3, HEIGHT, WIDTH))

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        trans1 = ScaleNew(size=(WIDTH, HEIGHT),
                                     interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST))
        trans2 = Normalize(mean=(INPUT_MEAN, (0,)), std=(INPUT_STD, (1,)))

        img = raw_img[240:, :, :]
        img = np.expand_dims(img, 0)
        img = trans1(img)
        img = trans2(img)
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

        cnt = 0
        for num in range(4):
            prob_map = (output[0][num + 1] * 255).astype(int)
            if cnt == 0:
                out_img = prob_map
            else:
                out_img += prob_map
            cnt += 1

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_img)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, HEIGHT*2, WIDTH)
    else:
        writer = None

    output_buffer = np.zeros((HEIGHT*2,WIDTH,3))
    output_buffer = output_buffer.astype(np.uint8)

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        trans = Normalize(mean=(INPUT_MEAN, (0,)), std=(INPUT_STD, (1,)))

        # resize with keep aspect
        frame,resized_img = webcamera_utils.adjust_frame_size(frame, HEIGHT, WIDTH)

        img = np.expand_dims(resized_img, 0)
        img = trans(img)
        img = np.array(img).transpose(0, 3, 1, 2)

        output, output_exist = net.run(img)
        output = softmax(output, axis=1)

        cnt = 0
        for num in range(4):
            prob_map = (output[0][num + 1] * 255).astype(int)
            if cnt == 0:
                out_img = prob_map
            else:
                out_img += prob_map
            cnt += 1

        out_img = np.array(out_img, dtype=np.uint8)

        # create output img
        output_buffer[0:HEIGHT,0:WIDTH,:] = resized_img
        output_buffer[HEIGHT:HEIGHT*2,0:WIDTH,0] = out_img
        output_buffer[HEIGHT:HEIGHT*2,0:WIDTH,1] = out_img
        output_buffer[HEIGHT:HEIGHT*2,0:WIDTH,2] = out_img

        cv2.imshow('output', output_buffer)

        # save results
        if writer is not None:
            writer.write(out_img)

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
