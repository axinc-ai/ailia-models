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

IMAGE_PATH = './bottle_000.png'
SAVE_IMAGE_PATH = './output.png'
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
    help='process with augmentation.'
)
parser.add_argument(
    '-an', '--aug_num', type=int, default=5,
    help='specify the amplification number of augmentation.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================



def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def plot_fig(file_list, test_imgs, scores, anormal_scores, gt_imgs, threshold, savepath):
    num = len(file_list)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        image_path = file_list[i]
        img = test_imgs[i]
        if not args.aug:
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

        fig_img.suptitle("Input : " + image_path + "  Anomaly score : " + str(anormal_scores[i]))
        logger.info("Anomaly score : " + str(anormal_scores[i]))

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

        if ('.' in savepath.split('/')[-1]):
            savepath_tmp = get_savepath(savepath, image_path, ext='.png')
        else:
            filename_tmp = image_path.split('/')[-1]
            ext_tmp = '.' + filename_tmp.split('.')[-1]
            filename_tmp = filename_tmp.replace(ext_tmp, '.png')
            savepath_tmp = '%s/%s' % (savepath, filename_tmp)
        logger.info(f'saved at : {savepath_tmp}')
        fig_img.savefig(savepath_tmp, dpi=100)
        plt.close()


def train_from_image(net, params):
    batch_size = int(args.batch_size)

    random.seed(args.seed)
    idx = random.sample(range(0, params["t_d"]), params["d"])

    params["idx"] = idx

    batch_size = int(args.batch_size)
    train_dir = args.train_dir
    train_outputs = training(net, params, batch_size, train_dir, args.aug, args.aug_num, logger)

    return train_outputs

def infer_one_image(net, params, train_outputs, dist_list, i_img, batch_size, test_imgs, gt_type_dir, gt_imgs, idx):
    # prepare input data
    imgs = []
    logger.info('from (%s ~ %s) ' %
                (args.input[i_img],
                    args.input[min(len(args.input) - 1,
                                i_img + batch_size)]))
    for image_path in args.input[i_img:i_img + batch_size]:
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img)
        test_imgs.append(img[0])
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
                if gt_img is not None:
                    gt_img = gt_img[0, [0]]
                else:
                    gt_img = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE))

        gt_imgs.append(gt_img)

    # countup N
    #N += len(imgs)

    imgs = np.vstack(imgs)

    logger.debug(f'input images shape: {imgs.shape}')
    net.set_input_shape(imgs.shape)

    # inference
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

    test_outputs = OrderedDict([
        ('layer1', []), ('layer2', []), ('layer3', [])
    ])
    for key, name in zip(test_outputs.keys(), params["feat_names"]):
        test_outputs[key].append(net.get_blob_data(name))
    for k, v in test_outputs.items():
        test_outputs[k] = v[0]

    embedding_vectors = postprocess(test_outputs)

    # randomly select d dimension
    embedding_vectors = embedding_vectors[:, idx, :, :]

    # reshape 2d pixels to 1d features
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(B, C, H * W)

    # calculate distance matrix
    dist_tmp = np.zeros([B, (H * W)])
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_tmp[:, i] = dist
    dist_list.append(dist_tmp)
    return B, C, H, W 


def infer_from_image(net, params, train_outputs):
    batch_size = 1#int(args.batch_size)
    random.seed(args.seed)
    idx = random.sample(range(0, params["t_d"]), params["d"])

    params["idx"] = idx

    gt_type_dir = args.gt_dir if args.gt_dir else None
    test_imgs = []
    gt_imgs = []
    angle_list = []
    pad_h_list = []
    pad_w_list = []

    N = 0
    dist_list = []
    for i_img in range(0, len(args.input), batch_size):
        B, C, H, W = infer_one_image(net, params, train_outputs, dist_list, i_img, batch_size, test_imgs, gt_type_dir, gt_imgs, idx)
        N = N + 1

    dist_list = np.vstack(dist_list)
    dist_list = dist_list.reshape(N, H, W)

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
    anormal_scores = np.zeros((score_map.shape[0]))
    for i in range(score_map.shape[0]):
        anormal_scores[i] = score_map[i].max()

    if args.threshold is None:
        # get optimal threshold
        if not args.aug:
            gt_mask = np.asarray(gt_imgs)
        else:
            gt_mask = np.asarray(gt_imgs[:int(len(gt_imgs) / args.aug_num)])
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


def recognize_from_image(net, params):
    if args.feat:
        logger.info('loading train set feature from: %s' % args.feat)
        with open(args.feat, 'rb') as f:
            train_outputs = pickle.load(f)
        logger.info('loaded.')
    else:
        train_outputs = train_from_image(net, params)
    infer_from_image(net, params, train_outputs)

def main():
    # model settings
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

    # create param
    params = {
        "feat_names": feat_names,
        "t_d": t_d,
        "d": d,
    }

    # create net instance
    net = ailia.Net(model_path, weight_path, env_id=args.env_id)

    # check input
    if len(args.input) == 0:
        logger.error("Input file not found")
        return

    recognize_from_image(net, params)


if __name__ == '__main__':
    main()
