import time
import argparse
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

import ailia

sys.path.append('../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402

from pose_resnet_util import get_final_preds, get_affine_transform, compute


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'balloon.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 256

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Simple Baselines for Human Pose Estimation and Tracking.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
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
MODEL_NAME = 'pose_resnet_50_256x192'
WEIGHT_PATH = f'{MODEL_NAME}.onnx'
MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/pose_resnet/'


# ======================
# Utils
# ======================
def plot_images(title, images, tile_shape):
    assert images.shape[0] <= (tile_shape[0] * tile_shape[1])
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    plt.title(title)
    grid = ImageGrid(fig, 111,  nrows_ncols=tile_shape)
    for i in range(images.shape[1]):
        grd = grid[i]
        grd.imshow(images[0, i])


# ======================
# Main functions
# ======================

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # prepare input data
    original_img = cv2.imread(args.input)

    detections = compute(net,original_img)
    threshold = 0.3

    for i in range(ailia.POSE_KEYPOINT_CNT):
        x = detections.points[i].x
        y = detections.points[i].y
        prob = detections.points[i].score

        if prob > threshold:
            circle_size = 2
            cv2.circle(original_img, (int(x), int(y)), circle_size,
                       (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(
                original_img,
                "{}".format(i),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                lineType=cv2.LINE_AA
            )

    cv2.imshow("Keypoints", original_img)
    cv2.imwrite(args.savepath, original_img)

    #channels = output.shape[1]#, paf.shape[1])
    #cols = 8
    #plot_images("output", output, tile_shape=(
    #    (int)((channels+cols-1)/cols), cols))
    #plt.show()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
