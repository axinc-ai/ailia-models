import os
import sys
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'resnet50_scratch.caffemodel'
MODEL_PATH = 'resnet50_scratch.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vggface2/'

IMAGE_PATH_1 = 'couple_a.jpg'
IMAGE_PATH_2 = 'couple_c.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MEAN = np.array([131.0912, 103.8827, 91.4953])  # to normalize input image
THRESHOLD = 1.00  # VGGFace2 predefined value 1~1.24
SLEEP_TIME = 3  # for video input mode


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Determine if the person is the same based on VGGFace2'
)
parser.add_argument(
    '-i', '--inputs', metavar='IMAGE',
    nargs=2,
    default=[IMAGE_PATH_1, IMAGE_PATH_2],
    help='Two image paths for calculating the face match.'
)
parser.add_argument(
    '-v', '--video', metavar=('VIDEO', 'IMAGE'),
    nargs=2,
    default=None,
    help='Determines whether the face in the video file specified by VIDEO ' +
         'and the face in the image file specified by IMAGE are the same. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def distance(feature1, feature2):
    norm1 = np.sqrt(np.sum(np.abs(feature1**2)))
    norm2 = np.sqrt(np.sum(np.abs(feature2**2)))
    dist = feature1/norm1-feature2/norm2
    l2_norm = np.sqrt(np.sum(np.abs(dist**2)))
    return l2_norm


def load_and_preprocess(img_path):
    img = load_image(
        img_path,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
        gen_input_ailia=False
    )
    return preprocess(img)


def preprocess(img, input_is_bgr=False):
    if input_is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize image
    input_data = (img.astype(np.float) - MEAN)
    input_data = input_data.transpose((2, 0, 1))
    input_data = input_data[np.newaxis, :, :, :]
    return input_data


# ======================
# Main functions
# ======================
def compare_images():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    features = []

    # prepare input data
    for j, img_path in enumerate(args.inputs):
        input_data = load_and_preprocess(img_path)

        # inference
        print('Start inference...')
        if args.benchmark and j == 0:
            # Bench mark mode is only for the first image
            print('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                _ = net.predict(input_data)
                end = int(round(time.time() * 1000))
                print(f'\tailia processing time {end - start} ms')
        else:
            _ = net.predict(input_data)

        blob = net.get_blob_data(net.find_blob_index_by_name('conv5_3'))
        features.append(blob)

    # get result
    fname1 = os.path.basename(args.inputs[0])
    fname2 = os.path.basename(args.inputs[1])
    dist = distance(features[0], features[1])
    print(f'{fname1} vs {fname2} = {dist}')

    if dist < THRESHOLD:
        print('Same person')
    else:
        print('Not same person')

    print('Script finished successfully.')


def compare_videoframe_image():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # img part
    fname = args.video[1]
    input_data = load_and_preprocess(fname)
    _ = net.predict(input_data)
    i_feature = net.get_blob_data(net.find_blob_index_by_name('conv5_3'))

    # video part
    if args.video[0] == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        _, resized_frame = adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = preprocess(resized_frame, input_is_bgr=True)

        # inference
        _ = net.predict(input_data)
        v_feature = net.get_blob_data(net.find_blob_index_by_name('conv5_3'))

        # show result
        dist = distance(i_feature, v_feature)
        print('=============================================================')
        print(f'{os.path.basename(fname)} vs video frame = {dist}')

        if dist < THRESHOLD:
            print('Same person')
        else:
            print('Not same person')
        cv2.imshow('frame', resized_frame)
        time.sleep(SLEEP_TIME)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        compare_videoframe_image()
    else:
        # image mode
        compare_images()


if __name__ == '__main__':
    main()
