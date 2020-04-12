import sys
import time
import argparse

import numpy as np
import cv2

import ailia
# import original modules
sys.path.append('../util')
from utils import check_file_existance
from model_utils import check_and_download_models
from image_utils import load_image
from webcamera_utils import adjust_frame_size


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Real-time hair segmentation model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH, 
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +\
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-n', '--normal',
    action='store_false',
    help='By default, the optimized model is used, but with this option, ' +\
    'you can switch to the normal (not optimized) model'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)

args = parser.parse_args()


# ======================
# Parameters 2
# ======================
NOT_OPT_MODEL = args.normal
if NOT_OPT_MODEL:
    WEIGHT_PATH = 'hair_segmentation.onnx'
else:
    WEIGHT_PATH = "hair_segmentation.opt.onnx"
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/hair_segmentation/"


# ======================
# Utils
# ======================
def transfer(image, mask):
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask

    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)
    return dst


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    src_img = cv2.imread(args.input)
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    input_data = input_data[np.newaxis, :, :, :]

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape(input_data.shape)
    
    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(input_data)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    pred = preds_ailia.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    dst = transfer(src_img, pred)
    cv2.imwrite(args.savepath, dst)


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    flag_set_shape = False

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

        input_image, input_data = adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB) / 255.0
        input_data = input_data[np.newaxis, :, :, :]

        if not flag_set_shape:
            net.set_input_shape(input_data.shape)
            flag_set_shape = True
        
        preds_ailia = net.predict(input_data)
        pred = preds_ailia.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        dst = transfer(input_image, pred)
        cv2.imshow('frame', dst)

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
