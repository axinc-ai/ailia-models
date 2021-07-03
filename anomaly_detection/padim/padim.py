import os
import sys
import time
from collections import OrderedDict
import random
import pickle

import numpy as np
import cv2
from PIL import Image
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

from padim_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_RESNET18_PATH = 'resnet18.onnx'
MODEL_RESNET18_PATH = 'resnet18.onnx.prototxt'
WEIGHT_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx'
MODEL_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'

IMAGE_PATH = 'bottle_000.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_RESIZE = 256
IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('PaDiM model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch', default='resnet18', choices=('resnet18', 'wide_resnet50_2'),
    help='arch model.'
)
parser.add_argument(
    '-f', '--feat', metavar="PICKLE_FILE", default=None,
    help='train set feature pkl files.'
)
parser.add_argument(
    '-bs', '--batch_size', default=32,
    help='batch size.'
)
parser.add_argument(
    '-tr', '--train_dir', metavar="DIR", default="./train",
    help='directory of the train files.'
)
parser.add_argument(
    '-gt', '--gt_dir', metavar="DIR", default="./gt_masks",
    help='directory of the ground truth mask files.'
)
parser.add_argument(
    '--seed', type=int, default=1024,
    help='random seed'
)
parser.add_argument(
    '-th', '--threshold', type=float, default=None,
    help='threshold'
)
parser.add_argument(
    '-ag', '--aug', action='store_true',
    help='execute augmentation.'
)
parser.add_argument(
    '-an', '--aug_num', type=int, default=5,
    help='specify the amplification number of augmentation.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img, mask=False):
    h, w = img.shape[:2]
    size = IMAGE_RESIZE
    crop_size = IMAGE_SIZE

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
    else:
        img = img / 255

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def preprocess_aug(img, mask=False, angle_range=[-10, 10], 
                   angle=None, pad_h=None, pad_w=None):
    h, w = img.shape[:2]
    size = IMAGE_RESIZE
    crop_size = IMAGE_SIZE

    # resize
    if h > w:
        size = (size, int(size * h / w))
    else:
        size = (int(size * w / h), size)
    img = np.array(Image.fromarray(img).resize(
        size, resample=Image.ANTIALIAS if not mask else Image.NEAREST))

    # random rotate
    if (angle is None):
        angle = np.random.randint(angle_range[0], angle_range[0] + 1)
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    if (mask is True):
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
    img = cv2.warpAffine(src=img,
                         M=rot_mat,
                         dsize=(w, h),
                         borderMode=cv2.BORDER_REPLICATE,
                         flags=interpolation)

    # random crop
    if (pad_h is None):
        pad_h = np.random.randint(0, (h - crop_size))
    if (pad_w is None):
        pad_w = np.random.randint(0, (w - crop_size))
    img = img[pad_h:pad_h + crop_size, pad_w:pad_w + crop_size, :]

    # normalize
    if not mask:
        img = normalize_image(img.astype(np.float32), 'ImageNet')
    else:
        img = img / 255

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img, angle, pad_h, pad_w


def postprocess(outputs):
    # Embedding concat
    embedding_vectors = outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, outputs[layer_name])

    return embedding_vectors


def get_train_outputs(net, create_net, params):
    if args.feat:
        logger.info('loading train set feature from: %s' % args.feat)
        with open(args.feat, 'rb') as f:
            train_outputs = pickle.load(f)
        logger.info('loaded.')
        return train_outputs

    batch_size = args.batch_size

    train_dir = args.train_dir
    train_imgs = sorted([
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
    ])
    if len(train_imgs) == 0:
        logger.error("train images not found in '%s'" % train_dir)
        sys.exit(-1)

    logger.info('extract train set features')

    if args.aug:
        aug_num = args.aug_num
    else:
        aug_num = 1
    mean = None
    N = 0
    for i_aug in range(aug_num):
        for i_img in range(0, len(train_imgs), batch_size):
            # prepare input data
            imgs = []
            for image_path in train_imgs[i_img:i_img + batch_size]:
                logger.info(image_path)
                img = load_image(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                if (not args.aug):
                    img = preprocess(img)
                else:
                    img, _, _, _ = preprocess_aug(img)
                imgs.append(img)

            # countup N
            N += len(imgs)

            imgs = np.vstack(imgs)

            logger.debug(f'input images shape: {imgs.shape}')
            if create_net:
                net = create_net()
            net.set_input_shape(imgs.shape)

            _ = net.predict(imgs)

            train_outputs = OrderedDict([
                ('layer1', []), ('layer2', []), ('layer3', [])
            ])
            for key, name in zip(train_outputs.keys(), params["feat_names"]):
                train_outputs[key].append(net.get_blob_data(name))
            for k, v in train_outputs.items():
                train_outputs[k] = v[0]

            embedding_vectors = postprocess(train_outputs)

            # randomly select d dimension
            idx = params['idx']
            embedding_vectors = embedding_vectors[:, idx, :, :]

            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.shape
            embedding_vectors = embedding_vectors.reshape(B, C, H * W)

            # initialize mean and covariance matrix
            if (mean is None):
                mean = np.zeros((C, H * W), dtype=np.float32)
                cov = np.zeros((C, C, H * W), dtype=np.float32)
                I = np.identity(C)

            # add up mean and covariance matrix
            mean += np.sum(embedding_vectors, axis=0)
            for i in range(H * W):
                # https://github.com/numpy/numpy/blob/v1.21.0/numpy/lib/function_base.py#L2324-L2543
                m = embedding_vectors[:, :, i]
                m = m - (mean[:, [i]].T / N)
                cov[:, :, i] += m.T @ m
    # devide mean by N
    mean = mean / N
    # devide covariance by N-1, and calculate inverse
    for i in range(H * W):
        cov[:, :, i] = (cov[:, :, i] / (N - 1)) + 0.01 * I

    train_outputs = [mean, cov]

    # save learned distribution
    train_feat_file = "%s.pkl" % os.path.basename(train_dir)
    logger.info('saving train set feature to: %s ...' % train_feat_file)
    with open(train_feat_file, 'wb') as f:
        pickle.dump(train_outputs, f)
    logger.info('saved.')

    return train_outputs


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def plot_fig(file_list, test_imgs, scores, anormal_scores, gt_imgs, threshold, save_dir):
    num = len(file_list)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        image_path = file_list[i]
        img = test_imgs[i]
        img = denormalization(img)
        gt = gt_imgs[i]
        gt = gt.transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)

        fig_img.suptitle("Input : "+image_path+"  Anomaly score : "+str(anormal_scores[i]))
        logger.info("Anomaly score : "+str(anormal_scores[i]))

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        savepath = get_savepath(save_dir, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        fig_img.savefig(savepath, dpi=100)
        plt.close()


def recognize_from_image(net, create_net, params):
    batch_size = args.batch_size

    random.seed(args.seed)
    idx = random.sample(range(0, params["t_d"]), params["d"])

    params["idx"] = idx
    train_outputs = get_train_outputs(net, create_net, params)

    gt_type_dir = args.gt_dir if args.gt_dir else None
    test_imgs = []
    gt_imgs = []

    # input image loop
    test_outputs = OrderedDict([
        ('layer1', []), ('layer2', []), ('layer3', [])
    ])
    if args.aug:
        aug_num = args.aug_num
    else:
        aug_num = 1
    for i_aug in range(aug_num):
        for i in range(0, len(args.input), batch_size):
            # prepare input data
            imgs = []
            for image_path in args.input[i:i + batch_size]:
                logger.info(image_path)
                img = load_image(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                if (not args.aug):
                    img = preprocess(img)
                else:
                    img, angle, pad_h, pad_w = preprocess_aug(img)
                imgs.append(img)

                # ground truth
                gt_img = None
                if gt_type_dir:
                    fname = os.path.splitext(os.path.basename(image_path))[0]
                    gt_fpath = os.path.join(gt_type_dir, fname + '_mask.png')
                    if os.path.exists(gt_fpath):
                        gt_img = load_image(gt_fpath)
                        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2RGB)
                        if (not args.aug):
                            gt_img = preprocess(gt_img, mask=True)
                        else:
                            gt_img, _, _, _ = preprocess_aug(gt_img, angle=angle,
                                                             pad_h=pad_h, pad_w=pad_w)
                        gt_img = np.mean(gt_img, axis=1, keepdims=True)

                gt_img = gt_img[0] if gt_img is not None else np.zeros((1, IMAGE_SIZE, IMAGE_SIZE))

                test_imgs.append(img[0])
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

            for key, name in zip(test_outputs.keys(), params["feat_names"]):
                test_outputs[key].append(net.get_blob_data(name))

    logger.info('postprocessing...')

    for k, v in test_outputs.items():
        test_outputs[k] = np.vstack(v)

    embedding_vectors = postprocess(test_outputs)

    # randomly select d dimension
    embedding_vectors = embedding_vectors[:, idx, :, :]

    # calculate distance matrix
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(B, C, H * W)
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    score_map = np.asarray([
        np.array(Image.fromarray(s).resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        ) for s in dist_list
    ])

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    # Calculated anormal score
    anormal_scores=np.zeros((score_map.shape[0]))
    for i in range(score_map.shape[0]):
        anormal_scores[i]=score_map[i].max()

    if args.threshold is None:
        # get optimal threshold
        gt_mask = np.asarray(gt_imgs)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        logger.info('Optimal threshold: %f' % threshold)
    else:
        threshold = args.threshold

    plot_fig(args.input, test_imgs, scores, anormal_scores, gt_imgs, threshold, args.savepath)

    logger.info('Script finished successfully.')


def main():
    info = {
        "resnet18": (
            WEIGHT_RESNET18_PATH, MODEL_RESNET18_PATH,
            ("140", "156", "172"), 448, 100),
        "wide_resnet50_2": (
            WEIGHT_WIDE_RESNET50_2_PATH, MODEL_WIDE_RESNET50_2_PATH,
            ("356", "398", "460"), 1792, 550),
    }
    # model files check and download
    weight_path, model_path, feat_names, t_d, d = info[args.arch]
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    params = {
        "feat_names": feat_names,
        "t_d": t_d,
        "d": d,
    }

    def _create_net():
        return ailia.Net(model_path, weight_path, env_id=args.env_id)

    # net initialize
    if True:
        create_net = _create_net
        net = None
    else:
        create_net = None
        net = _create_net()

    # check input
    if len(args.input)==0:
        logger.error("Input file not found")
        return

    recognize_from_image(net, create_net, params)


if __name__ == '__main__':
    main()
