import sys
import time
import platform
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402


# ======================
# PARAMETERS
# ======================
# MODEL_PATH = "gazeml_elg_i180x108_n64.onnx.prototxt"
# WEIGHT_PATH = "gazeml_elg_i180x108_n64.onnx"
# OUTPUT_BLOB_NAME = "import/hourglass/hg_3/after/hmap/conv/BiasAdd:0"

WEIGHT_PATH = 'gazeml_elg_i60x36_n32.onnx'
MODEL_PATH = 'gazeml_elg_i60x36_n32.onnx.prototxt'
OUTPUT_BLOB_NAME = "import/hourglass/hg_2/after/hmap/conv/BiasAdd:0"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gazeml/"

IMAGE_PATH = 'eye.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 36  # 108
IMAGE_WIDTH = 60  # 180

THRESHOLD = 0.1
SCALE = 1.2


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='gaze estimation model'
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
def plot_on_image(img, preds_ailia):
    for i in range(preds_ailia.shape[3]):
        probMap = preds_ailia[0, :, :, i]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (img.shape[1] * point[0]) / preds_ailia.shape[2] * SCALE
        y = (img.shape[0] * point[1]) / preds_ailia.shape[1] * SCALE
        color = (0, 255, 255)
        if i >= 8:
            color = (255, 0, 0)
        if i >= 16:
            color = (0, 0, 255)
        if prob > THRESHOLD:
            cv2.circle(
                img,
                (int(x), int(y)),
                3,
                color,
                thickness=-1,
                lineType=cv2.FILLED
            )
        # print(f'[DEBUG]  x: {int(x):3d}\ty: {int(y):3d}\tprob:{prob}')
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = cv2.imread(args.input)
    img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        normalize_type='None'
    )
    img = cv2.equalizeHist(img)
    if platform.system() == 'Darwin':  # For Mac OS (FP16)
        data = img[np.newaxis, np.newaxis, :, :] / 255.0 - 0.5
    else:
        data = img[np.newaxis, np.newaxis, :, :] / 127.5 - 1.0
    eyeI = np.concatenate((data, data), axis=0)
    eyeI = eyeI.reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

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
            preds_ailia = net.predict(eyeI)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(eyeI)
        
    preds_ailia = net.get_blob_data(
        net.find_blob_index_by_name(OUTPUT_BLOB_NAME)
    )

    # postprocessing
    img = plot_on_image(org_img, preds_ailia)
    cv2.imwrite(args.savepath, img)
    print('Script finished successfully.')


def recognize_from_video():
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

        # prepare frame
        img, resized_img = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        data = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        data = cv2.equalizeHist(data)
        if platform.system() == 'Darwin':
            data = data[np.newaxis, np.newaxis, :, :] / 255.0 - 0.5
        else:
            data = data[np.newaxis, np.newaxis, :, :] / 127.5 - 1.0
        eyeI = np.concatenate((data, data), axis=0)
        eyeI = eyeI.reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

        # inference
        preds_ailia = net.predict(eyeI)
        preds_ailia = net.get_blob_data(
            net.find_blob_index_by_name(OUTPUT_BLOB_NAME)
        )

        # postprocessing
        img = plot_on_image(img, preds_ailia)
        cv2.imshow('frame', img)

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
