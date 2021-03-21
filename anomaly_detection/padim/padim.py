import os
import sys
import time
from collections import OrderedDict
import random
import pickle

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from padim_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx'
MODEL_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'

IMAGE_PATH = 'img.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('PaDiM model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-f', '--feat', metavar="PICKLE_FILE", default=None,
    help='train set feature pkl files.'
)
parser.add_argument(
    '-tr', '--train_dir', metavar="DIR", default="./train",
    help='directory of the train files.'
)
parser.add_argument(
    '-gt', '--gt_dir', metavar="DIR", default="./gt_masks",
    help='directory of the ground truth mask files.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img, mask=False):
    h, w = img.shape[:2]
    size = 256
    crop_size = 224

    # resize
    if h > w:
        size = (size, int(size * h / w))
    else:
        size = (int(size * w / h), size)
    img = np.array(Image.fromarray(img).resize(
        size, resample=Image.ANTIALIAS if not mask else Image.NEAREST))

    # center crop
    h, w = img.shape[:2]
    pad_h = (h - crop_size) // 2
    pad_w = (w - crop_size) // 2
    img = img[pad_h:pad_h + crop_size, pad_w:pad_w + crop_size, :]

    # normalize
    if not mask:
        img = normalize_image(img.astype(np.float32), 'ImageNet')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def postprocess(outputs):
    # Embedding concat
    embedding_vectors = outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, outputs[layer_name])

    return embedding_vectors


def get_train_outputs(net, create_net, idx):
    if args.feat:
        logger.info('loading train set feature from: %s' % args.feat)
        with open(args.feat, 'rb') as f:
            train_outputs = pickle.load(f)
        logger.info('loaded.')
        return train_outputs

    train_dir = args.train_dir
    train_imgs = sorted([
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

    train_outputs = OrderedDict([
        ('layer1', []), ('layer2', []), ('layer3', [])
    ])
    batch_size = 32

    logger.info('extract train set features')

    for i in range(0, len(train_imgs), batch_size):
        # prepare input data
        imgs = []
        for image_path in train_imgs[i:i + batch_size]:
            logger.info(image_path)
            img = load_image(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img)
            imgs.append(img)

        imgs = np.vstack(imgs)

        logger.debug(f'input images shape: {imgs.shape}')
        if create_net:
            net = create_net()
        net.set_input_shape(imgs.shape)

        _ = net.predict(imgs)

        for key, name in zip(train_outputs.keys(), ("356", "398", "460")):
            train_outputs[key].append(net.get_blob_data(name))

    for k, v in train_outputs.items():
        train_outputs[k] = np.vstack(v)

    embedding_vectors = postprocess(train_outputs)

    # randomly select d dimension
    embedding_vectors = embedding_vectors[:, idx, :, :]

    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(B, C, H * W)
    mean = np.mean(embedding_vectors, axis=0)
    cov = np.zeros((C, C, H * W), dtype=np.float32)
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * I

    train_outputs = [mean, cov]

    # save learned distribution
    train_feat_file = "%s.pkl" % os.path.basename(train_dir)
    logger.info('saving train set feature to: %s ...' % train_feat_file)
    with open(train_feat_file, 'wb') as f:
        pickle.dump(train_outputs, f)
    logger.info('saved.')

    return train_outputs


def recognize_from_image(net, create_net):
    batch_size = 32
    t_d = 1792
    d = 550

    random.seed(1024)
    idx = random.sample(range(0, t_d), d)

    train_outputs = get_train_outputs(net, create_net, idx)

    test_outputs = OrderedDict([
        ('layer1', []), ('layer2', []), ('layer3', [])
    ])

    gt_type_dir = args.gt_dir if args.gt_dir else None

    test_imgs = args.input
    gt_imgs = []

    # input image loop
    for i in range(0, len(test_imgs), batch_size):
        # prepare input data
        imgs = []
        for image_path in test_imgs[i:i + batch_size]:
            logger.info(image_path)
            img = load_image(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img)
            imgs.append(img)

            # ground truth
            gt_img = None
            if gt_type_dir:
                fname = os.path.splitext(os.path.basename(image_path))[0]
                gt_fpath = os.path.join(gt_type_dir, fname + '_mask.png')
                if os.path.exists(gt_fpath):
                    gt_img = load_image(gt_fpath)
                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2RGB)
                    gt_img = preprocess(gt_img, mask=True)
                    gt_img = np.mean(gt_img, axis=1, keepdims=True)

            gt_imgs.append(gt_img)

        imgs = np.vstack(imgs)

        logger.debug(f'input images shape: {imgs.shape}')
        if create_net:
            net = create_net()
        net.set_input_shape(imgs.shape)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                _ = net.predict(imgs)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            _ = net.predict(imgs)

        for key, name in zip(test_outputs.keys(), ("356", "398", "460")):
            test_outputs[key].append(net.get_blob_data(name))

    for k, v in test_outputs.items():
        test_outputs[k] = np.vstack(v)

    for k, v in test_outputs.items():
        print("%s--" % k, v)
        print("%s--" % k, v.shape)

        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, xx)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    capture = get_capture(args.video)

    save_h, save_w = IMAGE_HEIGHT, IMAGE_WIDTH
    output_frame = np.zeros((save_h, save_w * 2, 3))

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    input_shape_set = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocess(img)

        # predict
        if (not input_shape_set):
            net.set_input_shape(img.shape)
            input_shape_set = True
        output = net.predict(img)

        hand_scoremap = output[0]
        hand_scoremap = np.argmax(hand_scoremap, 2) * 128
        res_img = hand_scoremap.astype("uint8")
        res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)

        output_frame[:, save_w:save_w * 2, :] = res_img
        output_frame[:, 0:save_w, :] = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        output_frame = output_frame.astype("uint8")

        cv2.imshow('frame', output_frame)

        # save results
        if writer is not None:
            writer.write(output_frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    info = {
        "wide_resnet50_2": (WEIGHT_WIDE_RESNET50_2_PATH, MODEL_WIDE_RESNET50_2_PATH),
    }
    # model files check and download
    weight_path, model_path = info["wide_resnet50_2"]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    def _create_net():
        return ailia.Net(model_path, weight_path, env_id=args.env_id)

    # net initialize
    if True:
        create_net = _create_net
        net = None
    else:
        create_net = None
        net = _create_net()

    if args.video is not None:
        # video mode
        recognize_from_video(net, create_net)
    else:
        # image mode
        recognize_from_image(net, create_net)


if __name__ == '__main__':
    main()
