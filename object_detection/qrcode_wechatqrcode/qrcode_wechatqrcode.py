# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import cv2 
import sys
import argparse
import numpy as np

from qrcode_wechatqrcode_utils import WeChatQRCode

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


WEIGHT_DETCT_PATH = 'detect.caffemodel'
MODEL_DETECT_PATH = 'detect.prototxt'
WEIGHT_SR_PATH = 'sr.caffemodel'
MODEL_SR_PATH = 'sr.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/qrcode_wechatqrcode/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'



def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Yolov2 model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)

def visualize(image, res, points, points_color=(0, 255, 0), text_color=(0, 255, 0), fps=None):
    output = image.copy()
    h, w, _ = output.shape

    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    fontScale = 0.5
    fontSize = 1
    for r, p in zip(res, points):
        p = p.astype(np.int32)
        for _p in p:
            cv2.circle(output, _p, 10, points_color, -1)

        qrcode_center_x = int((p[0][0] + p[2][0]) / 2)
        qrcode_center_y = int((p[0][1] + p[2][1]) / 2)

        text_size, baseline = cv2.getTextSize(r, cv2.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = qrcode_center_x - int(text_size[0] / 2)
        text_y = qrcode_center_y - int(text_size[1] / 2)
        cv2.putText(output, '{}'.format(r), (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)

    return output



def recognize_from_image():
    # net initialize
    model = WeChatQRCode(MODEL_DETECT_PATH,
        WEIGHT_DETCT_PATH,
        MODEL_SR_PATH,
        WEIGHT_SR_PATH)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                res, points = model.infer(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            res, points = model.infer(img)

        # plot result
        result = visualize(img, res, points)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, result)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    model = WeChatQRCode(MODEL_DETECT_PATH,
        WEIGHT_DETCT_PATH,
        MODEL_SR_PATH,
        WEIGHT_SR_PATH)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        res, points = model.infer(frame)
        result = visualize(frame, res, points)
        cv2.imshow('frame', result)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download

    #check_and_download_models(WEIGHT_DETCT_PATH, MODEL_DETECT_PATH, REMOTE_PATH)
    #check_and_download_models(WEIGHT_SR_PATH, MODEL_SR_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()


