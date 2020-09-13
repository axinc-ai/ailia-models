import sys
import time
import argparse
from collections import OrderedDict

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402C

# ======================
# Parameters
# ======================

WEIGHT_YOLOV3_MODANET_PATH = 'yolov3-modanet.onnx'
MODEL_YOLOV3_MODANET_PATH = 'yolov3-modanet.onnx.prototxt'
WEIGHT_YOLOV3_DF2_PATH = 'yolov3-df2.onnx'
MODEL_YOLOV3_DF2_PATH = 'yolov3-df2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

DATASETS_MODEL_PATH = OrderedDict([
    ('modanet', [WEIGHT_YOLOV3_MODANET_PATH, MODEL_YOLOV3_MODANET_PATH]),
    ('df2', [WEIGHT_YOLOV3_DF2_PATH, MODEL_YOLOV3_DF2_PATH])
])

IMAGE_PATH = '0000003.jpg'
SAVE_IMAGE_PATH = 'output.png'

MODANET_CATEGORY = [
    "bag", "belt", "boots", "footwear", "outer", "dress", "sunglasses",
    "pants", "top", "shorts", "skirt", "headwear", "scarf/tie"
]
DF2_CATEGORY = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "shorts", "trousers", "skirt", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress"
]
THRESHOLD = 0.5
IOU = 0.4
DETECTION_WIDTH = 416

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Clothing detection model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-d', '--dataset', metavar='TYPE', choices=DATASETS_MODEL_PATH,
    default=list(DATASETS_MODEL_PATH.keys())[0],
    help='Type of dataset to train the model. Allowed values are {}.'.format(', '.join(DATASETS_MODEL_PATH))
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-dw', '--detection_width',
    default=DETECTION_WIDTH,
    help='The detection width and height for yolo. (default: 416)'
)
args = parser.parse_args()

weight_path, model_path = DATASETS_MODEL_PATH[args.dataset]


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = load_image(args.input)
    # img = cv2.imread(args.input)  # BGR
    # img = cv2.resize(img, (416, 416))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # x = img
    print(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
    x = np.zeros([h, w + pad1 + pad2, 3], dtype=np.uint8) \
        if w <= h else np.zeros([h + pad1 + pad2, w, 3], dtype=np.uint8)
    x[:, :, :] = 127.5
    if w <= h:
        x[:, pad1:pad1 + w, :] = img
    else:
        x[pad1:pad1 + h, :, :] = img
    x = cv2.resize(x, (416, 416))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

    dataset_category = MODANET_CATEGORY if args.dataset == 'modanet' else DF2_CATEGORY

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Detector(
        model_path,
        weight_path,
        len(dataset_category),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id
    )
    # if int(args.detection_width) != DETECTION_WIDTH:
    #     detector.set_input_shape(int(args.detection_width), int(args.detection_width))
    detector.set_input_shape(416, 416)

    # inferece
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            detector.compute(img, THRESHOLD, IOU)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        detector.compute(x, THRESHOLD, IOU)

    print("----")
    count = detector.get_object_count()
    print(count)
    for idx in range(count):
        obj = detector.get_object(idx)
        print("obj---", obj)

    # plot result
    res_img = plot_results(detector, img, dataset_category)
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video():
    dataset_category = MODANET_CATEGORY if args.dataset == 'modanet' else DF2_CATEGORY

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Detector(
        model_path,
        weight_path,
        len(dataset_category),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id
    )
    if int(args.detection_width) != DETECTION_WIDTH:
        detector.set_input_shape(int(args.detection_width), int(args.detection_width))

    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)
        res_img = plot_results(detector, frame, dataset_category, False)
        cv2.imshow('frame', res_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    # check_and_download_models(weight_path, model_path, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
