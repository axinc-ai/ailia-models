import os
import sys
import time
import argparse

from matplotlib import pyplot as plt

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame, get_capture  # noqa: E402


# TODO Upgrade Model
# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'resnet_facial_feature.onnx'
MODEL_PATH = 'resnet_facial_feature.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/resnet_facial_feature/"

IMAGE_PATH = 'test.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 226
IMAGE_WIDTH = 226


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='kaggle facial keypoints.'
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
# Utils
# ======================
def gen_img_from_predsailia(input_data, preds_ailia):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(input_data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
    points = np.vstack(np.split(preds_ailia, 15)).T * 113 + 113
    ax.plot(points[0], points[1], 'o', color='red')
    return fig


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        gen_input_ailia=True
    )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(img)[0]
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(img)[0]

    # postprocess
    fig = gen_img_from_predsailia(img, preds_ailia)
    fig.savefig(args.savepath)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        input_image, input_data = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, data_rgb=False
        )

        # inference
        preds_ailia = net.predict(input_data)[0]

        # postprocessing
        fig = gen_img_from_predsailia(input_data, preds_ailia)
        fig.savefig('tmp.png')
        img = cv2.imread('tmp.png')
        cv2.imshow('frame', img)

    capture.release()
    cv2.destroyAllWindows()
    os.remove('tmp.png')
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
