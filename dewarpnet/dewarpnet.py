import sys
import time
import argparse

import cv2
import numpy as np

import ailia
# import original modules
sys.path.append('../util')
from utils import check_file_existance
from model_utils import check_and_download_models
from image_utils import load_image
from webcamera_utils import preprocess_frame


# ======================
# PARAMETERS
# ======================
BM_WEIGHT_PATH = "bm_model.onnx"
WC_WEIGHT_PATH = "wc_model.onnx"
BM_MODEL_PATH = "bm_model.onnx.prototxt"
WC_MODEL_PATH = "wc_model.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/dewarpnet/"

IMAGE_PATH = 'test.png'
SAVE_IMAGE_PATH = 'result.png'

WC_IMG_HEIGHT = 256
WC_IMG_WIDTH = 256
BM_IMG_HEIGHT = 128
BM_IMG_WIDTH = 128


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='DewarpNet is a model for document image unwarping.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
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
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def grid_sample(img, grid):
    height, width, c = img.shape
    output = np.zeros_like(img)
    grid[:, :, 0] = (grid[:, :, 0] + 1) * (width-1) / 2
    grid[:, :, 1] = (grid[:, :, 1] + 1) * (height-1) / 2
    # TODO speed up here
    for h in range(height):
        for w in range(width):
            h_ = int(grid[h, w, 1])
            w_ = int(grid[h, w, 0])
            output[h, w] = img[h_, w_]
    return output


def unwarp(img, bm):
    w, h = img.shape[0], img.shape[1]
    bm = bm.transpose(1, 2, 0)
    bm0 = cv2.blur(bm[:, :, 0], (3, 3))
    bm1 = cv2.blur(bm[:, :, 1], (3, 3))
    bm0 = cv2.resize(bm0, (h, w))
    bm1 = cv2.resize(bm1, (h, w))
    bm = np.stack([bm0, bm1], axis=-1)
    img = img.astype(float) / 255.0
    res = grid_sample(img, bm)
    return res


# ======================
# Main functions
# ======================
def unwarp_from_image():
    org_img = cv2.imread(args.input)
    img = load_image(
        args.input,
        (WC_IMG_HEIGHT, WC_IMG_WIDTH),
        normalize_type='255',
        gen_input_ailia=True
    )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    bm_net = ailia.Net(BM_MODEL_PATH, BM_WEIGHT_PATH, env_id=env_id)
    wc_net = ailia.Net(WC_MODEL_PATH, WC_WEIGHT_PATH, env_id=env_id)

    # compute exectuion time
    for i in range(5):
        start = int(round(time.time() * 1000))
    
        wc_output = wc_net.predict(img)[0]
        pred_wc = np.clip(wc_output, 0, 1.0).transpose(1, 2, 0)
        bm_input = cv2.resize(
            pred_wc, (BM_IMG_WIDTH, BM_IMG_HEIGHT)
        ).transpose(2, 0, 1)
        bm_input = np.expand_dims(bm_input, 0)
        outputs_bm = bm_net.predict(bm_input)[0]
        uwpred = unwarp(org_img, outputs_bm)  # This is not on GPU!
    
        end = int(round(time.time() * 1000))
        print("ailia processing time {} ms".format(end-start))

    cv2.imwrite(args.savepath, uwpred * 255)
    print('Script finished successfully.')


def unwarp_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    bm_net = ailia.Net(BM_MODEL_PATH, BM_WEIGHT_PATH, env_id=env_id)
    wc_net = ailia.Net(WC_MODEL_PATH, WC_WEIGHT_PATH, env_id=env_id)

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

        org_image, input_data = preprocess_frame(
            frame, WC_IMG_HEIGHT, WC_IMG_WIDTH, normalize_type='255'
        )
        
        # inference
        wc_output = wc_net.predict(input_data)[0]
        pred_wc = np.clip(wc_output, 0, 1.0).transpose(1, 2, 0)
        bm_input = cv2.resize(
            pred_wc, (BM_IMG_WIDTH, BM_IMG_HEIGHT)
        ).transpose(2, 0, 1)
        bm_input = np.expand_dims(bm_input, 0)
        outputs_bm = bm_net.predict(bm_input)[0]
        uwpred = unwarp(org_image, outputs_bm)  # This is not on GPU!

        cv2.imshow('frame', uwpred)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(BM_WEIGHT_PATH, BM_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WC_WEIGHT_PATH, WC_MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        unwarp_from_video()
    else:
        # image mode
        unwarp_from_image()


if __name__ == '__main__':
    main()
    
