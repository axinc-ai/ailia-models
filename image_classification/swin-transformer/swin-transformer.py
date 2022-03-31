import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'swin-transformer_tiny_patch4_window7_224.onnx'
MODEL_PATH = 'swin-transformer_tiny_patch4_window7_224.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/swin-transformer/'

IMAGE_PATH = 'example/ILSVRC2012_val_00026142.JPEG'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Swin Transformer for Image Classification', IMAGE_PATH, None)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def get_labels():
    labels = {}
    with open('label_table.txt', 'r') as f:
         data = f.readlines()
         for d in data:
             d = d.replace('\n', '').split('\t')
             labels[int(d[0])] = [d[1], d[2]]
    return labels


def recognize_from_image():
    # get labels
    labels = get_labels()

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for i, image_path in enumerate(args.input):
        # prepare input data
        logger.info(image_path)
        im = cv2.imread(image_path)

        # resize
        h, w = im.shape[0], im.shape[1]
        if w > h:
            new_h = 224
            new_w = int(224*(w/h))
        else:
            new_h = int(224*(h/w))
            new_w = 224
        im = cv2.resize(im, dsize=(new_w, new_h))

        # center crop
        size = 224
        x = new_w/2 - size/2
        y = new_h/2 - size/2
        im = im[int(y):int(y+size), int(x):int(x+size), :]
        im = im.transpose((2, 0, 1))
        im = im[np.newaxis, :, :, :]
        input = im

        # predict
        output = net.predict(input)

        # postprocess
        topk = 3
        output = output[0]
        unsorted_max_indices = np.argpartition(-output, topk)[:topk]
        output = output[unsorted_max_indices]
        indices = np.argsort(-output)
        max_k_indices = unsorted_max_indices[indices]

        # show
        print('==============================================================')
        for idx in range(topk):
            print(f'+ idx={idx}')
            print(f'  category={max_k_indices[idx]}['
                  f'{labels[max_k_indices[idx]][1]} ]')
            print(f'  prob={output[indices[idx]]}')

    logger.info('Script finished successfully.')


def recognize_from_video():
    # get labels
    labels = get_labels()

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        im = frame

        # resize
        h, w = im.shape[0], im.shape[1]
        if w > h:
            new_h = 224
            new_w = int(224*(w/h))
        else:
            new_h = int(224*(h/w))
            new_w = 224
        im = cv2.resize(im, dsize=(new_w, new_h))

        # center crop
        size = 224
        x = new_w/2 - size/2
        y = new_h/2 - size/2
        im = im[int(y):int(y+size), int(x):int(x+size), :]
        im = im.transpose((2, 0, 1))
        im = im[np.newaxis, :, :, :]
        input = im

        # predict
        output = net.predict(input)

        # postprocess
        output = np.argmax(output, axis=1)

        # show
        logger.info('class = {}'.format(labels[output[0]][1]))

    capture.release()
    cv2.destroyAllWindows()

    logger.info('Script finished successfully.')


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
