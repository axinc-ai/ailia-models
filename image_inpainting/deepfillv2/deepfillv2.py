import sys
import time
import os

import numpy as np
import cv2
import onnx
import onnxruntime as rt

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from deepfillv2_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_CELEBA256_PATH = 'deepfillv2_celeba_256x256.onnx'
MODEL_CELEBA256_PATH = 'deepfillv2_celeba_256x256.onnx.prototxt'
WEIGHT_CELEBA512_PATH = 'deepfillv2_celeba_512x512.onnx'
MODEL_CELEBA512_PATH = 'deepfillv2_celeba_512x512.onnx.prototxt'
WEIGHT_CELEBA1024_PATH = 'deepfillv2_celeba_1024x1024.onnx'
MODEL_CELEBA1024_PATH = 'deepfillv2_celeba_1024x1024.onnx.prototxt'

WEIGHT_PLACES256_PATH = 'deepfillv2_places_256x256.onnx'
MODEL_PLACES256_PATH = 'deepfillv2_places_256x256.onnx.prototxt'
WEIGHT_PLACES512_PATH = 'deepfillv2_places_512x512.onnx'
MODEL_PLACES512_PATH = 'deepfillv2_places_512x512.onnx.prototxt'
WEIGHT_PLACES1024_PATH = 'deepfillv2_places_1024x1024.onnx'
MODEL_PLACES1024_PATH = 'deepfillv2_places_1024x1024.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deepfillv2/'

IMAGE_PATH = 'paris-streetview_001.png'
SAVE_IMAGE_PATH = 'result.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('deepfillv2 model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model', default="places", choices=("places", "celeba"),
    help='mask type'
)
parser.add_argument(
    '-mt', '--mask_type', default="stroke", choices=("rect", "stroke"),
    help='mask type'
)
parser.add_argument(
    '--seed', type=int, default=1,
    help='random seed for random stroke mask'
)
parser.add_argument(
    '--random_mask', type=int, default=0,
    help='using random stroke mask'
)
parser.add_argument(
    '--mask_shape', type=str, default='128,128',
    help='given mask parameters: h,w'
)
parser.add_argument(
    '-rm', '--run_mode', default="ailia", choices=("ailia", "onnxruntime"),
    help='framework to run model'
)
parser.add_argument(
    '-ir', '--img_res', type=int, default=256, choices=(256, 512, 1024),
    help='mask type'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def pad(img, ds_factor=32, mode='reflect'):
    h, w = img.shape[:2]

    new_h = ds_factor * ((h - 1) // ds_factor + 1)
    new_w = ds_factor * ((w - 1) // ds_factor + 1)

    pad_h = new_h - h
    pad_w = new_w - w
    if new_h != h or new_w != w:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        img = np.pad(img, pad_width[:img.ndim], mode = mode)
    
    return img

def imnormalize(img, mean, std, to_rgb=True):
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

def recognize_from_image(net, img_shape):
    if args.random_mask:
        np.random.seed(args.seed)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare ground truth
        gt_img = load_image(image_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2BGR)
        gt_img = cv2.resize(gt_img, img_shape)

        # prepare mask
        if args.mask_type == 'rect':
            str_mask_shape = args.mask_shape.split(',')
            mask_shape = [int(x) for x in str_mask_shape]
            mask = generate_mask_rect(img_shape, mask_shape, args.random_mask)
        else:
            mask = generate_mask_stroke(
                im_size=(img_shape[0], img_shape[1]),
                parts=8, maxBrushWidth=24, maxLength=100, maxVertex=20)

        # mask = load_image("D:/ax/ailia-models/image_inpainting/deepfillv2/bbox_mask1.png")
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
        # mask = cv2.resize(mask, (256, 256))
        # mask = mask[:,:,:1]
        # mask[mask > 0] = 1.

        # prepare input data
        img_mask = gt_img * (1 - mask) + 255 * mask

        img = pad(gt_img.copy())
        mask = pad(mask)
        mean, std = np.array([127.5] * 3), np.array([127.5] * 3)
        img = imnormalize(img, mean, std, to_rgb=False)

        img = img * (1. - mask).astype('float32')

        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        img = np.transpose(img, (0, 3, 1, 2))
        mask = np.transpose(mask, (0, 3, 1, 2)) # N,C,H,W
        tmp_ones = np.ones_like(mask)
        data = np.concatenate((img, tmp_ones, mask), 1)

        logger.debug(f'input data shape: {data.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict(data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            if args.run_mode == 'ailia':
                # output = net.predict(data)
                output = net.run(data)
            elif args.run_mode == 'onnxruntime':
                output = net.run(None, {
                    'input': data,
                    })
        output = output[1] # The outputs are 2 images, 1st is coarse one, 2nd is the refined one (output name = "1802")
        
        fake_img = output * mask + img * (1. - mask)
        fake_img = np.transpose(fake_img[0], (1 ,2, 0))
        min_max = (-1, 1)
        fake_img = ((fake_img - min_max[0]) / (min_max[1] - min_max[0])*255.0).round()

        img_mask = img_mask.astype(np.uint8)
        mask = np.transpose(mask, (0, 2, 3, 1))
        mask = cv2.cvtColor(mask[0], cv2.COLOR_GRAY2BGR).astype(np.uint8) * 255.0
        res_img = np.hstack((img_mask, mask, fake_img, gt_img))

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')

def main():
    info = {
        ("celeba", 256): (WEIGHT_CELEBA256_PATH, MODEL_CELEBA256_PATH, (256, 256)),
        ("places", 256): (WEIGHT_PLACES256_PATH, MODEL_PLACES256_PATH, (256, 256)),
        ("places", 512): (WEIGHT_PLACES512_PATH, MODEL_PLACES512_PATH, (512, 512)),
        ("places", 1024): (WEIGHT_PLACES1024_PATH, MODEL_PLACES1024_PATH, (1024, 1024))
    }
    key = (args.model, args.img_res)
    if key not in info:
        logger.error("(MODEL = %s, IMG_RESOLUTION = %s) is unmatch." % key)
        logger.info("appropriate settings:\n"
                    "\t(MODEL = celeba, IMG_RESOLUTION = 256)\n"
                    "\t(MODEL = places, IMG_RESOLUTION = 256 or 512 or 1024)")
        sys.exit(-1)

    # model files check and download
    weight_path, model_path, img_shape = info[key]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # net initialize
    if args.run_mode == 'ailia':
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    elif args.run_mode == 'onnxruntime':
        # check by onnx
        onnx_model = onnx.load(weight_path)
        onnx.checker.check_model(onnx_model)
        net = rt.InferenceSession(weight_path)

    recognize_from_image(net, img_shape)


if __name__ == '__main__':
    main()
