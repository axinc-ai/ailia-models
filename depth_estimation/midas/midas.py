import os
import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer, \
    calc_adjust_fsize  # noqa: E402
from image_utils import normalize_image  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'midas.onnx'
MODEL_PATH = 'midas.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/midas/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'input_depth.png'
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
IMAGE_MULTIPLE_OF = 32


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MiDaS model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def constrain_to_multiple_of(x, min_val=0, max_val=None):
    y = (np.round(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if y < min_val:
        y = (np.ceil(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    return y


def midas_resize(image, target_height, target_width):
    # Resize while keep aspect ratio.
    h, w, _ = image.shape
    scale_height = target_height / h
    scale_width = target_width / w
    if scale_width < scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height
    new_height = constrain_to_multiple_of(
        scale_height * h, max_val=target_height
    )
    new_width = constrain_to_multiple_of(
        scale_width * w, max_val=target_width
    )

    return cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )


def midas_imread(image_path):
    if not os.path.isfile(image_path):
        logger.error(f'{image_path} not found.')
        sys.exit()
    image = cv2.imread(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, 'ImageNet')

    return midas_resize(image, IMAGE_HEIGHT, IMAGE_WIDTH)


def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = midas_imread(image_path)
        img = img.transpose((2, 0, 1))  # channel first
        img = img[np.newaxis, :, :, :]

        logger.debug(f'input image shape: {img.shape}')
        net.set_input_shape(img.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = net.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            result = net.predict(img)

        depth_min = result.min()
        depth_max = result.max()
        max_val = (2 ** 16) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (result - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out.transpose(1, 2, 0).astype("uint16"))
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
        save_h, save_w = calc_adjust_fsize(f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH)
        # save_w * 2: we stack source frame and estimated heatmap
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    input_shape_set = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        resized_img = midas_resize(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        resized_img = resized_img.transpose((2, 0, 1))  # channel first
        resized_img = resized_img[np.newaxis, :, :, :]

        if(not input_shape_set):
            net.set_input_shape(resized_img.shape)
            input_shape_set = True
        result = net.predict(resized_img)

        depth_min = result.min()
        depth_max = result.max()
        max_val = (2 ** 16) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (result - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        res_img = out.transpose(1, 2, 0).astype("uint16")
        cv2.imshow('depth', res_img)

        # save results
        if writer is not None:
            # FIXME: cannot save correctly...
            res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)
            writer.write(cv2.convertScaleAbs(res_img))

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
