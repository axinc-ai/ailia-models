import sys
import time
import argparse

import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage import io

import ailia
from illnet_utils import *

# import original modules
sys.path.append('../../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'illnet.onnx'
MODEL_PATH = 'illnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/illnet/'

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Illumination Correction Model'
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
def recognize_from_image():
    # prepare input data
    img = io.imread(args.input)
    img = preProcess(img)
    input_data = padCropImg(img)
    input_data = input_data.astype(np.float32) / 255.0

    ynum = input_data.shape[0]
    xnum = input_data.shape[1]

    preds_ailia = np.zeros((ynum, xnum, 128, 128, 3), dtype=np.float32)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))

            for j in range(ynum):
                for i in range(xnum):
                    patchImg = input_data[j, i]
                    patchImg = (patchImg - 0.5) / 0.5
                    patchImg = patchImg.transpose((2, 0, 1))
                    patchImg = patchImg[np.newaxis, :, :, :]
                    out = net.predict(patchImg)
                    out = out.transpose((0, 2, 3, 1))[0]
                    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                    preds_ailia[j, i] = out

            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        start = int(round(time.time() * 1000))

        for j in range(ynum):
            for i in range(xnum):
                patchImg = input_data[j, i]
                patchImg = (patchImg - 0.5) / 0.5
                patchImg = patchImg.transpose((2, 0, 1))
                patchImg = patchImg[np.newaxis, :, :, :]
                out = net.predict(patchImg)
                out = out.transpose((0, 2, 3, 1))[0]
                out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                preds_ailia[j, i] = out

        end = int(round(time.time() * 1000))

    # postprocessing
    resImg = composePatch(preds_ailia)
    resImg = postProcess(resImg)
    resImg.save(args.savepath)
    print('Script finished successfully.')


def recognize_from_video():
    # [WARNING] This is test impl
    print('[WARNING] This is test implementation')
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

        img = preProcess(frame)
        input_data = padCropImg(img)
        input_data = input_data.astype(np.float32) / 255.0

        ynum = input_data.shape[0]
        xnum = input_data.shape[1]

        preds_ailia = np.zeros((ynum, xnum, 128, 128, 3), dtype=np.float32)

        for j in range(ynum):
            for i in range(xnum):
                patchImg = input_data[j, i]
                patchImg = (patchImg - 0.5) / 0.5
                patchImg = patchImg.transpose((2, 0, 1))
                patchImg = patchImg[np.newaxis, :, :, :]
                out = net.predict(patchImg)
                out = out.transpose((0, 2, 3, 1))[0]
                out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                preds_ailia[j, i] = out

        resImg = composePatch(preds_ailia)
        resImg = postProcess(resImg)
        cv2.imshow('frame', img_as_ubyte(resImg))

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
