import os
import sys
import numpy as np
import cv2
import matplotlib

import ailia

matplotlib.use('TkAgg')

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger

import webcamera_utils
from detector_utils import (load_image, plot_results, reverse_letterbox,
                            write_predictions)
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models
from arg_utils import get_base_parser, get_savepath, update_parser

from lib.run import *

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ======================
# Parameters
# ======================
REMOTE_PATH_YOLO = 'https://storage.googleapis.com/ailia-models/yolox/'
REMOTE_PATH_POSE = 'https://storage.googleapis.com/ailia-models/strided_transformer_pose3d/'

# settings
WEIGHT_PATH_YOLO = "yolox_tiny.opt.onnx"
MODEL_PATH_YOLO = "yolox_tiny.opt.onnx.prototxt"

WEIGHT_PATH_POSE2D = "hrnet_w48-8ef0771d.opt.onnx"
MODEL_PATH_POSE2D = "hrnet_w48-8ef0771d.opt.onnx.prototxt"

WEIGHT_PATH_POSE3D = "no_refine_4365.opt.onnx"
MODEL_PATH_POSE3D = "no_refine_4365.opt.onnx.prototxt"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

# Default input size
HEIGHT = 416
WIDTH = 416

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Strided Transformer POSE3D model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)

# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = args.env_id
    net_yolo = ailia.Detector(
        MODEL_PATH_YOLO,
        WEIGHT_PATH_YOLO,
        80,
        format=ailia.NETWORK_IMAGE_FORMAT_BGR,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
        env_id=env_id)

    net_pose2d = ailia.Net(MODEL_PATH_POSE2D, WEIGHT_PATH_POSE2D, env_id=env_id)
    net_pose3d = ailia.Net(MODEL_PATH_POSE3D, WEIGHT_PATH_POSE3D, env_id=env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)

        # inference
        logger.info('Start inference...')
        image_pose2d, fig_pose3d, _ = detect(raw_img, net_yolo, net_pose2d, net_pose3d)

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite('2D_'+savepath, image_pose2d)
        plt.savefig('3D_'+savepath, dpi=200, format='png', bbox_inches='tight')

    logger.info('Script finished successfully.')

def recognize_from_video():
    env_id = args.env_id
    net_yolo = ailia.Net(MODEL_PATH_YOLO, WEIGHT_PATH_YOLO, env_id=env_id)
    net_pose2d = ailia.Net(MODEL_PATH_POSE2D, WEIGHT_PATH_POSE2D, env_id=env_id)
    net_pose3d = ailia.Net(MODEL_PATH_POSE3D, WEIGHT_PATH_POSE3D, env_id=env_id)

    capture = webcamera_utils.get_capture(args.video)

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

    frame_shown = False
    fig = plt.figure(figsize=(9.6, 5.4))
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        frame = cv2.resize(frame, dsize=(960, 640))
        image_pose2d, ax_pose3d, detect_flg = detect(frame, net_yolo, net_pose2d, net_pose3d, fig)

        if detect_flg:
            cv2.imshow("Demo", image_pose2d)
            plt.pause(.01)
            ax_pose3d.clear()

        if not plt.get_fignums():
            break

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
    check_and_download_models(WEIGHT_PATH_YOLO, MODEL_PATH_YOLO, REMOTE_PATH_YOLO)
    check_and_download_models(WEIGHT_PATH_POSE2D, MODEL_PATH_POSE2D, REMOTE_PATH_POSE)
    check_and_download_models(WEIGHT_PATH_POSE3D, MODEL_PATH_POSE3D, REMOTE_PATH_POSE)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()