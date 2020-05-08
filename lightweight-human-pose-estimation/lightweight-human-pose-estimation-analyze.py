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


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'balloon.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

ALGORITHM = ailia.POSE_ALGORITHM_LW_HUMAN_POSE


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Fast and accurate human pose 2D-estimation.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
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
MODEL_NAME = 'lightweight-human-pose-estimation'
if args.normal:
    WEIGHT_PATH = f'{MODEL_NAME}.onnx'
    MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
else:
    WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
    MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


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
    input_image = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
    )
    input_data = (input_image - 128) / 255.0
    input_data = input_data[np.newaxis, :, :, :].transpose((0, 3, 1, 2))

    for i in range(3):
        if(i == 1):
            net.set_profile_mode()
        start = int(round(time.time() * 1000))
        _ = net.predict(input_data)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    confidence = net.get_blob_data(net.find_blob_index_by_name("397"))
    paf = net.get_blob_data(net.find_blob_index_by_name("400"))
    points = []
    threshold = 0.1

    for i in range(confidence.shape[1]):
        probMap = confidence[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (src_img.shape[1] * point[0]) / confidence.shape[3]
        y = (src_img.shape[0] * point[1]) / confidence.shape[2]

        if prob > threshold:
            circle_size = 4
            cv2.circle(src_img, (int(x), int(y)), circle_size,
                       (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(
                src_img,
                "{}".format(i),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                1,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                src_img,
                ""+str(prob),
                (int(x), int(y+circle_size)),
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

    channels = max(confidence.shape[1], paf.shape[1])
    cols = 8

    plot_images("paf", paf, tile_shape=((int)((channels+cols-1)/cols), cols))
    plot_images("confidence", confidence, tile_shape=(
        (int)((channels+cols-1)/cols), cols))

    plt.show()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
