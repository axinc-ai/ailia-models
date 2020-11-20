import sys
import time
import argparse

import cv2
from PIL import Image
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'colorizer.onnx'
MODEL_PATH = 'colorizer.onnx.prototxt'
REMOTE_PATH = ''

IMAGE_PATH = 'imgs/ansel_adams1.jpg'
SAVE_IMAGE_PATH = 'imgs_out/ansel_adams1_output.jpg'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Colorful Image Colorization model'
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
def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np


def preprocess(img_rgb_orig, resample=3):
    img_rgb_rs = np.asarray(
        Image.fromarray(img_rgb_orig).resize((IMAGE_WIDTH, IMAGE_HEIGHT), 
        resample=resample)
    )

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_lab_orig = img_lab_orig[:,:,0][None,None,:,:]
    img_lab_rs = img_lab_rs[:,:,0][None,None,:,:]

    return (img_lab_orig, img_lab_rs)


def post_process(out, img_lab_orig):
    HW_orig = img_lab_orig.shape[2:]
    out_ab_orig = cv2.resize(out.transpose(2, 3, 1, 0).squeeze(), (HW_orig[1],  HW_orig[0]), interpolation = cv2.INTER_LINEAR)
    out_ab_orig = np.expand_dims(out_ab_orig.transpose(2, 0, 1), 0)
    out_lab_orig = np.concatenate([img_lab_orig, out_ab_orig], 1)
    
    out_img = color.lab2rgb(out_lab_orig[0,...].transpose((1, 2, 0)))
    return out_img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = load_img(IMAGE_PATH)
    print(f'input image shape: {img.shape}')
    (img_lab_orig, img_lab_rs) = preprocess(img)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            out = net.predict({'input.1':img_lab_rs})[0]
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        out = net.predict({'input.1':img_lab_rs})[0]

    out_img = post_process(out, img_lab_orig)
    plt.imsave(args.savepath, out_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

    capture = get_capture(args.video)
    while(True):
        ret, img = capture.read()
        (img_lab_orig, img_lab_rs) = preprocess(img)
        out = net.predict({'input.1':img_lab_rs})[0]
        out_img = post_process(out, img_lab_orig)
        cv2.imshow('frame', out_img)

        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

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
