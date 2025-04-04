import math
import os
import sys
import time

import ailia
import cv2
import numpy as np



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

logger = getLogger(__name__)

# ======================
# Parameters
# ======================


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('yolox model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model_name',
    default='yolox_s',
    help='[yolox_nano, yolox_tiny, yolox_s, yolox_m, yolox_l,'
         'yolox_darknet, yolox_x]'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
parser.add_argument(
    '-th', '--threshold',
    default=SCORE_THR, type=float,
    help='The detection threshold for yolo. (default: '+str(SCORE_THR)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=NMS_THR, type=float,
    help='The detection iou for yolo. (default: '+str(NMS_THR)+')'
)
parser.add_argument(
    '-dt', '--detector',
    action='store_true',
    help='Use detector API (require ailia SDK 1.2.9).'
)
parser.add_argument(
    '-dw', '--detection_width',
    default=-1, type=int,
    help='The detection width and height for yolo. (default: auto)'
)
parser.add_argument(
    '-dh', '--detection_height',
    default=-1, type=int,
    help='The detection height and height for yolo. (default: auto)'
)
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".opt.onnx"
MODEL_PATH = MODEL_NAME + ".opt.onnx.prototxt"

HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]

# ======================
# Main functions
# ======================
def recognize_from_video(detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    # if args.write_prediction is not None:
    #     frame_count = 0
    #     frame_digit = int(math.log10(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 1)
    #     video_name = os.path.splitext(os.path.basename(args.video))[0]

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        raw_img = frame
        if args.detector:
            detector.compute(raw_img, args.threshold, args.iou)
            res_img = plot_results(detector, raw_img, COCO_CATEGORY)
            detect_object = detector
        else:
            img, ratio = preprocess(raw_img, (HEIGHT, WIDTH))
            output = detector.run(img[None, :, :, :])
            predictions = postprocess(output[0], (HEIGHT, WIDTH))[0]
            detect_object = predictions_to_object(predictions, raw_img, ratio, args.iou, args.threshold)
            detect_object = reverse_letterbox(detect_object, raw_img, (raw_img.shape[0], raw_img.shape[1]))
            res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

        # write prediction
        if args.write_prediction is not None:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, detect_object, frame, category=COCO_CATEGORY, file_type=ext)
            frame_count += 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id
    if args.detector:
        detector = ailia.Detector(
                MODEL_PATH,
                WEIGHT_PATH,
                len(COCO_CATEGORY),
                format=ailia.NETWORK_IMAGE_FORMAT_BGR,
                channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
                range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
                algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
                env_id=env_id)
        if args.detection_width!=-1 or args.detection_height!=-1:
            detector.set_input_shape(args.detection_width,args.detection_height)
    else:
        detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        if args.detection_width!=-1 or args.detection_height!=-1:
            global WIDTH,HEIGHT
            WIDTH=args.detection_width
            HEIGHT=args.detection_height
            detector.set_input_shape((1,3,HEIGHT,WIDTH))

    if args.video is not None:
        # video mode
        recognize_from_video(detector)
    # else:
    #     # image mode
    #     if args.profile:
    #         detector.set_profile_mode(True)
    #     recognize_from_image(detector)
    #     if args.profile:
    #         print(detector.get_summary())


if __name__ == '__main__':
    main()
