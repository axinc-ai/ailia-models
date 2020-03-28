import sys
import time
import argparse

import numpy as np

import ailia

# import original modules
sys.path.append('../util')
from model_utils import check_and_download_models
from image_utils import load_image


# ======================
# PARAMETERS
# ======================
IMG_PATH_1 = 'correct_pair_1.jpg'  # Base image
IMG_PATH_2 = 'correct_pair_2.jpg'  # Correct Pair image
IMG_PATH_3 = 'incorrect.jpg'     # Incorrect Pair image
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# the threshold was calculated by the `test_performance` function in `test.py`
# of the original repository
THRESHOLD = 0.25572845  

WEIGHT_PATH = 'arcface.onnx'
MODEL_PATH = 'arcface.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/arcface/"


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Determine if the person is the same from two facial images.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
    nargs=2,
    default=[IMG_PATH_1, IMG_PATH_2],
    help='Two iamge paths for calculating the face match'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def prepare_input_data(image_path):
    # (ref: https://github.com/ronghuaiyang/arcface-pytorch/issues/14)
    # use origin image and fliped image to infer,
    # and concat the feature as the final feature of the origin image.
    image = load_image(
        image_path,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        normalize_type='None'
    )
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image = image / 127.5 - 1.0  # normalize
    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# ======================
# Main function
# ======================
def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # prepare input data
    imgs_1 = prepare_input_data(args.input[0])
    imgs_2 = prepare_input_data(args.input[1])
    imgs = np.concatenate([imgs_1, imgs_2], axis=0)
    
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # compute execution time
    print('Start inference...')
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(imgs)
        end = int(round(time.time() * 1000))
        print("ailia processing time {} ms".format(end - start))

    # postprocessing
    fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
    fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
    sim = cosin_metric(fe_1, fe_2)

    print(
        'Similarity of (' + args.input[0] + ', ' + args.input[1] + f') : {sim}'
    )
    if THRESHOLD > sim:
        print('They are not the same face!')
    else:
        print('They are the same face!')
    

if __name__ == "__main__":
    main()
