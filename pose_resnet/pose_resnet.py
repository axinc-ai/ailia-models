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

from inference import get_final_preds, get_affine_transform


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
    src_img = cv2.imread(args.input)
    src_img = cv2.resize(src_img,(IMAGE_WIDTH,IMAGE_HEIGHT))

    w=src_img.shape[1]
    h=src_img.shape[0]
    print(w,h)

    image_size = [IMAGE_WIDTH, IMAGE_HEIGHT]

    print(image_size)

    input_data = src_img
    cv2.imwrite("affine.png", input_data)

    center=np.array([w/2, h/2], dtype=np.float32)
    scale = np.array([1, 1], dtype=np.float32)

    #BGR format
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    input_data = (input_data/255.0 - mean) / std
    input_data = input_data[np.newaxis, :, :, :].transpose((0, 3, 1, 2))

    #print(input_data)
    print(input_data.shape)

    for i in range(2):
        if(i == 1):
            net.set_profile_mode()
        start = int(round(time.time() * 1000))
        output = net.predict(input_data)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')
    
    preds, maxvals = get_final_preds(output, [center], [scale])

    points = []
    threshold = 0.1

    for i in range(preds.shape[1]):
        x=preds[0,i,0]
        y=preds[0,i,1]
        prob=maxvals[0,i,0]

        if prob > threshold:
            circle_size = 2
            cv2.circle(src_img, (int(x), int(y)), circle_size,
                       (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(
                src_img,
                "{}".format(i),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                lineType=cv2.LINE_AA
            )

            points.append((int(x), int(y)))
        else:
            points.append(None)

    cv2.imshow("Keypoints", src_img)
    cv2.imwrite(args.savepath, src_img)

    print(output.shape)

    #channels = output.shape[1]#, paf.shape[1])
    #cols = 8
    #plot_images("output", output, tile_shape=(
    #    (int)((channels+cols-1)/cols), cols))
    #plt.show()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
