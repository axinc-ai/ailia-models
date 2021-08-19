import sys
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter

import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

from detectron_utils import *

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SNEAKER_PATH = 'object_detection_3d_sneakers.onnx'
MODEL_SNEAKER_PATH = 'object_detection_3d_sneakers.onnx.prototxt'
WEIGHT_CHAIR_PATH = 'object_detection_3d_chair.onnx'
MODEL_CHAIR_PATH = 'object_detection_3d_chair.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/objectron-3d-object-detection-openvino/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGTH_SIZE = 640
IMAGE_WIDTH_SIZE = 480
THRESHOLD = 0.7

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'objectron-3d-object-detection-openvino',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-m', '--model', default='sneaker', choices=('sneaker', 'chair'),
    help='model type'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for detection.'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH_SIZE, IMAGE_HEIGTH_SIZE), cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def detect_peak(image, filter_size=3, order=0.5):
    local_max = maximum_filter(
        image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))

    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index


def decode(hm, displacements, threshold=0.8):
    hm = hm.reshape(hm.shape[2:])  # (40,30)
    peaks = detect_peak(hm)
    peakX = peaks[1]
    peakY = peaks[0]

    scaleX = hm.shape[1]
    scaleY = hm.shape[0]
    objs = []
    for x, y in zip(peakX, peakY):
        conf = hm[y, x]
        if conf < threshold:
            continue
        points = []
        for i in range(8):
            dx = displacements[0, i * 2, y, x]
            dy = displacements[0, i * 2 + 1, y, x]
            points.append((x / scaleX + dx, y / scaleY + dy))
        objs.append(points)

    return objs


def draw_detections(img, reg_detections, ids=None, rgb=True):
    # if image in RGB space --> convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if rgb else img

    """Draws detections and labels"""
    for kp, _id in zip(
            reg_detections, ids if ids else [''] * len(reg_detections)):
        if kp is not None and _id != 'ID -1':
            img = draw_kp(img, kp, normalized=False)

    if ids:
        for kp, _id in zip(reg_detections, ids):
            x0 = int(np.min(kp[1:, 0]))
            y0 = int(np.min(kp[1:, 1]))
            label_size, base_line = cv2.getTextSize(
                _id, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            y0 = max(y0, label_size[1])
            cv2.rectangle(
                img,
                (x0, y0 - label_size[1]), (x0 + label_size[0], y0 + base_line),
                (255, 255, 255), cv2.FILLED)
            cv2.putText(
                img, _id, (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return img


def transform_kp(kp: np.array, crop_cords: tuple):
    x0, y0, x1, y1 = crop_cords
    crop_shape = (x1 - x0, y1 - y0)
    kp[:, 0] = kp[:, 0] * crop_shape[0]
    kp[:, 1] = kp[:, 1] * crop_shape[1]
    kp[:, 0] += x0
    kp[:, 1] += y0

    return kp


# ======================
# Main functions
# ======================

def predict(net, img):
    threshold = args.threshold

    # initial preprocesses
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    hm, displacements, _ = output
    hm = hm.transpose(0, 3, 1, 2)
    displacements = displacements.transpose(0, 3, 1, 2)

    objs = decode(hm, displacements, threshold=threshold)

    return objs, hm


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                # Pose estimation
                start = int(round(time.time() * 1000))
                kps, hm = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            # inference
            kps, hm = predict(net, img)

        h, w = img.shape[:2]
        kps = np.asarray(kps).reshape(-1, 8, 2)
        kps = [transform_kp(kp, (0, 0, w, h)) for kp in kps]
        if 0 < len(kps):
            kps = np.insert(kps, 0, -100, axis=1)
        res_img = draw_detections(img, kps)

        # save results
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

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

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps, hm = predict(net, img)

        h, w = img.shape[:2]
        kps = np.asarray(kps).reshape(-1, 8, 2)
        kps = [transform_kp(obj, (0, 0, w, h)) for obj in kps]
        if 0 < len(kps):
            kps = np.insert(kps, 0, -100, axis=1)
        frame = draw_detections(frame, kps, rgb=False)

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
    dic_model = {
        'sneaker': (WEIGHT_SNEAKER_PATH, MODEL_SNEAKER_PATH),
        'chair': (WEIGHT_CHAIR_PATH, MODEL_CHAIR_PATH),
    }
    weight_path, model_path = dic_model[args.model]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    logger.info(f'env_id: {env_id}')

    # initialize
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
