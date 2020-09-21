import os
import sys
import time
import argparse

import numpy as np
import cv2
import onnxruntime

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402C
from image_utils import load_image, normalize_image  # noqa: E402C

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
parser = argparse.ArgumentParser(
    description='MiDaS model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
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
args = parser.parse_args()

# ======================
# Main functions
# ======================
def constrain_to_multiple_of(x, min_val = 0, max_val = None):
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
    new_height = constrain_to_multiple_of(scale_height * h, max_val = target_height)
    new_width = constrain_to_multiple_of(scale_width * w, max_val = target_width)

    return cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)


def midas_imread(image_path):
    if not os.path.isfile(image_path):
        print(f'[ERROR] {image_path} not found.')
        sys.exit()
    image = cv2.imread(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, 'ImageNet')

    return midas_resize(image, IMAGE_HEIGHT, IMAGE_WIDTH)


def recognize_from_image():
    # prepare input data
    img = midas_imread(args.input)
    img = img.transpose((2, 0, 1))  # channel first
    img = img[np.newaxis, :, :, :]
    print(f'input image shape: {img.shape}')
    
    # # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape(img.shape)

    # inferece
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            result = net.predict(img)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        result = net.predict(img)
    
    depth_min = result.min()
    depth_max = result.max()
    max_val = (2 ** 16) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (result - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    
    cv2.imwrite(args.savepath, out.transpose(1, 2, 0).astype("uint16"))    
    print('Script finished successfully.')


def recognize_from_video():
    # # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    input_shape_set = False
    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

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

        cv2.imshow('depth', out.transpose(1, 2, 0).astype("uint16"))

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


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
