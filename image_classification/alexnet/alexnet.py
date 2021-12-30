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

# logger
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

from PIL import Image


# ======================
# PARAMETERS
# ======================
MODEL_PATH  = "alexnet.onnx.prototxt"
WEIGHT_PATH = "alexnet.onnx"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/alexnet/"
IMAGE_PATH = "input/dog.jpg"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser("Alexnet is ", IMAGE_PATH, None,)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def _get_image(filename):
    input_image = Image.open(filename)
    input_batch = _preprocess_image(input_image)
    return input_batch


def _preprocess_image(img):
    if args.video is not None:
        img = Image.fromarray(img)

    # alternative to transforms.Resize(256)
    size = 256
    w = np.asarray(img).shape[1]
    h = np.asarray(img).shape[0]
    short, long = (w, h) if w <= h else (h, w)
    requested_new_short = size if isinstance(size, int) else size[0]
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    img = np.asarray(img)
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = np.array(img, mode_to_nptype.get('RGB', np.uint8))
    img = cv2.resize(img, (new_w, new_h))

    # alternative to transforms.CenterCrop(224)
    center_w = img.shape[1]/2
    center_h = img.shape[0]/2
    side = 224
    x = center_w - side/2
    y = center_h - side/2
    img = img[int(y):int(y+side), int(x):int(x+side), :]

    # alternative to transform.ToTensor()
    img = img.transpose((2, 0, 1))
    img = img / 255

    # alternative to transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.array(img)
    img = (img - mean[:, None, None]) / std[:, None, None]

    input_tensor = img
    input_batch = np.expand_dims(input_tensor, axis=0) # create a mini-batch as expected by the model
    return input_batch


def _get_prob(output, topk=5):
    prob = _softmax(output[0])
    idx = np.argsort(-prob)[:topk]
    y = prob[idx]
    topk_prob = y
    topk_catid = idx
    return topk_prob, topk_catid


def _softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    e_x = e_x / np.sum(e_x, axis=axis, keepdims=True)
    return e_x


def _get_labels():
    categories = []
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def recognize_from_image():
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    labels = _get_labels()

    # input image loop
    for i, image_path in enumerate(args.input):
        input_batch = _get_image(image_path)
        output = net.predict(input_batch)
        topk_prob, topk_catid = _get_prob(output)
        print('[Image_{}] {}'.format(i+1, image_path))
        for k in range(topk_prob.shape[0]):
            print('\t{} {}'.format(labels[topk_catid[k]], topk_prob[k]))

    logger.info('Script finished successfully.')


def recognize_from_video():
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    labels = _get_labels()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print('Video is not opened.')
        exit()
    f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    idx = -1
    while True:
        idx += 1
        ret, frame = cap.read()
        if not ret:
            break
        # predict
        if idx%(cap.get(cv2.CAP_PROP_FPS)*5) != 0:
            continue
        else:
            frame = cv2.resize(frame, dsize=(f_h, f_w))
            input_batch = _preprocess_image(frame)
            output = net.predict(input_batch)
            topk_prob, topk_catid = _get_prob(output)
            print('[Image] second: {}, frame: {}, path: {}'.format(idx//cap.get(cv2.CAP_PROP_FPS), idx, args.video))
            for k in range(topk_prob.shape[0]):
                print('\t{} {}'.format(labels[topk_catid[k]], topk_prob[k]))

    cap.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # recognize
    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
