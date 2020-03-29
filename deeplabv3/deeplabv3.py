import time
import os
import sys
import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

import ailia
from utils import *

# import original modules
sys.path.append('../util')
from model_utils import check_and_download_models
from image_utils import load_image, get_image_shape


# ======================
# PARAMETERS
# ======================
IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'output.png'
THRESHOLD = 160
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
CLASS_NUM = 21
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
assert CLASS_NUM == len(LABEL_NAMES), 'The number of labels is incorrect.'


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='DeepLab is a state-of-art deep learning model ' +\
    'for semantic image segmentation.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-c', '--camera',
    action='store_true',
    help='Running the model with the webcam image as input.'
)
parser.add_argument(
    '-n', '--normal',
    action='store_false',
    help='By default, the optimized model is used, but with this option, ' +\
    'you can switch to the normal model'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
args = parser.parse_args()


# ======================
# MODEL PARAMETERS
# ======================
if args.normal:
    MODEL_PATH = 'deeplabv3.opt.onnx.prototxt'
    WEIGHT_PATH = 'deeplabv3.opt.onnx'
else:
    MODEL_PATH = 'deeplabv3.onnx.prototxt'
    WEIGHT_PATH = 'deeplabv3.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deeplabv3/'


# ======================
# Main functions
# ======================
def segment_from_image():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    ailia_input_width = net.get_input_shape()[3]
    ailia_input_height = net.get_input_shape()[2]
    input_shape = [ailia_input_height, ailia_input_width]

    # prepare input data
    img = load_image(
        args.input, input_shape, normalize_type='127.5', gen_input_ailia=True
    )

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(img)[0]
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')        

    # postprocessing
    # original size
    seg_map = np.argmax(preds_ailia.transpose(1, 2, 0), axis=2)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)

    # save just segmented image (simple)
    # seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('seg_test.png', seg_image)  

    # save org_img, segmentation-map, segmentation-overlay
    org_img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
    org_img = cv2.resize(org_img, (seg_image.shape[1], seg_image.shape[0]))

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(org_img)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(org_img)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest'
    )
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.savefig('test.png')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.camera:
        # video mode
        segment_from_video()
    else:
        # image mode
        segment_from_image()


if __name__ == '__main__':
    main()
