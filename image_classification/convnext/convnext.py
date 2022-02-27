import time
import sys

import cv2
import numpy as np

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

# for ConvNeXt
import os


# ======================
# PARAMETERS 1
# ======================
IMAGE_PATH = "input.jpg"
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser("ConvNeXt is ", IMAGE_PATH, None,)
args = update_parser(parser)


# ==========================
# MODEL AND OTHER PARAMETERS
# ==========================
MODEL_PATH = "convnext_tiny_CIFAR-10.onnx.prototxt"
WEIGHT_PATH = "convnext_tiny_CIFAR-10.onnx"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/convnext/"


# ======================
# Main functions
# ======================
def _resize(img, size):
    h = img.shape[0]
    w = img.shape[1]
    short, long = (w, h) if w <= h else (h, w)
    requested_new_short = size if isinstance(size, int) else size[0]
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    img = cv2.resize(img, dsize=(new_w, new_h))
    return img

def _center_crop(img, w, h):
    x = img.shape[1]/2 - w/2
    y = img.shape[0]/2 - h/2
    img = img[int(y):int(y+h), int(x):int(x+w)]
    return img

def _preprocess(img):
    img = _resize(img, 224)
    img = _center_crop(img, 224, 224)
    img = img / 255
    img = img.transpose((2,0,1))
    img = img[np.newaxis, :, :, :]
    return img

def recognize_from_image(net):
    for image_path in args.input:
        # prepare input data
        if os.path.isfile(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            logger.error(f'{image_path} not found.')
            exit()

        # inference
        image = _preprocess(image)
        output = net.predict(image)
        pred_class = np.argmax(output)

        print('predicted class = {}({}), {}'.format(pred_class, CLASSES[pred_class], image_path))

    return

def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    i = 0
    while(True):
        i += 1
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        frame = _preprocess(frame)
        output = net.predict(frame)
        pred_class = np.argmax(output)

        print('predicted class = {}({}) frame = {}'.format(pred_class, CLASSES[pred_class], i))

    capture.release()
    cv2.destroyAllWindows()
    return

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    if args.video is not None: # video mode
        recognize_from_video(net)
    else: # image mode
        recognize_from_image(net)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
