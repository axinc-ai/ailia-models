import os
import sys
import time
import numpy as np
from glob import glob
from logging import getLogger

import ailia

# import local modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser # noqa: E402
from image_utils import load_image
from model_utils import check_and_download_models # noqa: E402


logger = getLogger(__name__)


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'facenet-pytorch', None, None 
)
parser.add_argument(
    '-d', '--dir', default='data',
    help='Directory path of input image files.'
)
parser.add_argument(
    '-w', '--weight', default='vggface2',
    choices=['vggface2', 'casia-webface'],
    help='You can choose the model weights.'
)
args = update_parser(parser)


# ======================
# Parameters
# ======================
WEIGHT_PATH = args.weight + '.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/facenet-pytorch/'

THRESHOLD = 1.0


# ======================
# Main Functions
# ======================
def preprocess(img_dir):
    files_path = os.path.join(img_dir, '*.*')
    filenames = []
    imgs = []

    for fp in glob(files_path):
        imgs.append(load_image(
            fp, (160, 160), True, '127.5', True
        ))
        filenames.append(fp.split(os.sep)[-1])

    return imgs, filenames


def predict(model, inputs):
    outputs = [model.predict(i) for i in inputs]
    return np.concatenate(outputs)


def postprocess(embeddings):
    pair_ids = []
    distances = []
    is_same = []

    for i in range(len(embeddings)):
        for j in range(i):
            pair_ids.append((j, i))

            dist = np.linalg.norm(embeddings[i]-embeddings[j])
            distances.append(dist)

            is_same.append(dist < THRESHOLD)

    return pair_ids, distances, is_same


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, None, None)

    # initialize
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # prepare data
    imgs, filenames = preprocess(args.dir)

    # inference
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            embeddings = predict(model, imgs)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end-start} ms')
    else:
        embeddings = predict(model, imgs)

    # calucate distances
    pair_ids, distances, is_same = postprocess(embeddings)

    # results
    for pair, dist, is_same in zip(pair_ids, distances, is_same):
        fn1, fn2 = filenames[pair[0]], filenames[pair[1]]
        logger.info(f'[Same face: {is_same}] Similarity of {fn1, fn2} is {dist:.3f}.')


if __name__ == '__main__':
    main()

