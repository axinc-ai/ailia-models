import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'G_ep300.onnx'
MODEL_PATH = 'G_ep300.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/beauty_gan/'

IMAGE_PATH = 'xfsy_0147.png'
IMAGE_MAKEUP_PATH = 'makeup_vFG48.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 256

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('BeautyGAN model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-im', '--image_makeup',
    default=IMAGE_MAKEUP_PATH, type=str, metavar='IMAGE',
    help='Makeup image.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    mean = np.array((0.5, 0.5, 0.5))
    std = np.array((0.5, 0.5, 0.5))

    h, w = img.shape[:2]
    if h > w:
        h = int(h / w * IMAGE_SIZE)
        w = IMAGE_SIZE
    else:
        w = int(w / h * IMAGE_SIZE)
        h = IMAGE_SIZE

    img = np.array(Image.fromarray(img).resize(
        (w, h),
        resample=Image.BILINEAR))

    if h > IMAGE_SIZE:
        p = (h - IMAGE_SIZE) // 2
        img = img[p:p + IMAGE_SIZE, :, :]
    elif w > IMAGE_SIZE:
        p = (w - IMAGE_SIZE) // 2
        img = img[:, p:p + IMAGE_SIZE, :]

    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def postprocess(output):
    output = (output.transpose((1, 2, 0)) + 1) / 2.0 * 255.0
    output = np.clip(output, 0, 255)
    img = output.astype(np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR

    return img


def recognize_from_image(net):
    img_B = load_image(args.image_makeup)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGRA2RGB)
    img_B = preprocess(img_B)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare grand truth
        img_A = load_image(image_path)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGRA2RGB)
        img_A = preprocess(img_A)

        logger.debug(f'input image shape: {img_A.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict({'img_A': img_A, 'img_B': img_B})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict({'img_A': img_A, 'img_B': img_B})

        fake_A, fake_B = output
        output = np.concatenate([img_A[0], img_B[0], fake_A[0], fake_B[0]], axis=2)
        res_img = postprocess(output)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
