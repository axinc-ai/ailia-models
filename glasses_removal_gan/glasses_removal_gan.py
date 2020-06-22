import os
import sys
import time
import argparse

from matplotlib import pyplot as plt

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = 'glasses_removal_gan.onnx'
MODEL_PATH = 'glasses_removal_gan.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/glasses_removal_gan/"

IMAGE_PATH = 'me.jpg'
SAVE_IMAGE_PATH = 'output.png'


# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='Glasses removal GAN based on SimGAN'
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
def process_image():
    # prepare input data
    img = load_image(
        args.input,
        (128, 128),
        rgb=False,
        normalize_type='255'
    )[None,...,None]

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
            preds_ailia = (net.predict(img)[0]*255).astype(np.uint8)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = (net.predict(img)[0]*255).astype(np.uint8)

    # postprocess
    cv2.imwrite(args.savepath, preds_ailia)
    print('Script finished successfully.')
    

def process_video():
    # net initialize
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

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (128, 128))[None,...,None]/255
 
        # inference
        preds_ailia = (net.predict(frame)[0]*255).astype(np.uint8)

        cv2.imshow('frame', preds_ailia)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
#     check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        process_video()
    else:
        # image mode
        process_image()


if __name__ == '__main__':
    main()
