from base64 import decode
from unicodedata import category
from unittest import result
import cv2 
import sys
import numpy as np
import time
import pyzbar.pyzbar as zbar

import ailia

from qrcode_wechatqrcode_utils import preprocess, postprocess, reverse_letterbox

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================
MODEL_PARAMS = {'detect_2021nov': {'input_shape': [384, 384], 'max_stride': 32, 'anchors':[
                    [12,16, 19,36, 40,28], [36,75, 76,55, 72,146], [142,110, 192,243, 459,401]
                    ]},
                }

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov7/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('QR detection model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model_name',
    default='detect_2021nov',
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".caffemodel"
MODEL_PATH = MODEL_NAME + ".prototxt"

HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]
STRIDE = MODEL_PARAMS[MODEL_NAME]['max_stride']
ANCHORS = MODEL_PARAMS[MODEL_NAME]['anchors']

def visualize(raw_img, detections):
    result_img = raw_img.copy()

    for d in detections:
        EXTRA_OFFSET = 10
        left = max(d.x - EXTRA_OFFSET, 0)
        top = max(d.y - EXTRA_OFFSET, 0)
        right = min(d.x + d.w + EXTRA_OFFSET, raw_img.shape[1])
        bottom = min(d.y + d.h + EXTRA_OFFSET, raw_img.shape[0])
        cropped = raw_img[top:bottom, left:right, :]

        decoded = zbar.decode(cropped)
        if len(decoded) > 0:
            text = decoded[0].data.decode()
            cv2.putText(result_img, text, (d.x, d.y + d.h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

        cv2.rectangle(result_img, (d.x, d.y), (d.x + d.w, d.y + d.h), (255, 0, 0))

    return result_img

def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        raw_img = cv2.imread(image_path)
        logger.debug(f'input image shape: {raw_img.shape}')

        img = preprocess(raw_img, (HEIGHT, WIDTH))

        # inference
        logger.info('Start inference...')
        res = None
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                res = net.run(img[None, None, :, :])
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            res = net.run(img[None, None, :, :])

        detections = postprocess(img, raw_img.shape, res)
        result_img = visualize(raw_img, detections)

        cv2.imshow("QR", result_img)
        cv2.waitKey()

        savepath = get_savepath(args.savepath, image_path)
        cv2.imwrite(savepath, result_img)
        logger.info(f'saved at : {savepath}')

    logger.info('Script finished successfully.')

def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, raw_frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        frame = preprocess(raw_frame, (HEIGHT, WIDTH))
        
        res = net.run(frame[None, None, :, :])
        detections = postprocess(frame, raw_frame.shape, res)
        result_frame = visualize(raw_frame, detections)

        cv2.imshow('frame', result_frame)

        # save results
        if writer is not None:
            writer.write(result_frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 1, WIDTH, HEIGHT))

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)

if __name__ == '__main__':
    main()
