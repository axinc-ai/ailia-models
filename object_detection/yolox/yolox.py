import numpy as np
import time
import os
import sys
import cv2
import math

from yolox_utils import preproc as preprocess
from yolox_utils import multiclass_nms, postprocess, predictions_to_object

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import reverse_letterbox, plot_results, write_predictions
import webcamera_utils

# logger
from logging import getLogger

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
MODEL_PARAMS = {'yolox_nano': {'input_shape': [416, 416]},
                'yolox_tiny': {'input_shape': [416, 416]},
                'yolox_s': {'input_shape': [640, 640]},
                'yolox_m': {'input_shape': [640, 640]},
                'yolox_l': {'input_shape': [640, 640]},
                'yolox_darknet': {'input_shape': [640, 640]},
                'yolox_x': {'input_shape': [640, 640]}}

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolox/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

SCORE_THR = 0.4
NMS_THR = 0.45

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('yolox model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--model_name',
    default='yolox_s',
    help='[yolox_nano, yolox_tiny, yolox_s, yolox_m, yolox_l,'
         'yolox_darknet, yolox_x]'
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
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
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".opt.onnx"
MODEL_PATH = MODEL_NAME + ".opt.onnx.prototxt"

HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]

# ======================
# Main functions
# ======================
def recognize_from_image():
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')
        img, ratio = preprocess(raw_img, (HEIGHT, WIDTH))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.run(img[None, :, :, :])
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            output = output = net.run(img[None, :, :, :])

        predictions = postprocess(output[0], (HEIGHT, WIDTH))[0]
        detect_object = predictions_to_object(predictions, raw_img, ratio, args.iou, args.threshold)
        detect_object = reverse_letterbox(detect_object, raw_img, (raw_img.shape[0], raw_img.shape[1]))
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction:
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(pred_file, detect_object, raw_img, COCO_CATEGORY)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

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

    if args.write_prediction:
        frame_count = 0
        frame_digit = int(math.log10(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 1)
        video_name = os.path.splitext(os.path.basename(args.video))[0]

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        raw_img = frame
        img, ratio = preprocess(raw_img, (HEIGHT, WIDTH))
        output = net.run(img[None, :, :, :])
        predictions = postprocess(output[0], (HEIGHT, WIDTH))[0]
        detect_object = predictions_to_object(predictions, raw_img, ratio, args.iou, args.threshold)
        detect_object = reverse_letterbox(detect_object, raw_img, (raw_img.shape[0], raw_img.shape[1]))
        res_img = plot_results(detect_object, raw_img, COCO_CATEGORY)
        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img)

        # write prediction
        if args.write_prediction:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(pred_file, detect_object, frame, COCO_CATEGORY)
            frame_count += 1

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
