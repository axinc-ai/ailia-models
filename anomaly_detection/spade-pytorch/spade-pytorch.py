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

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'wide_resnet50_2.onnx'
MODEL_PATH = 'wide_resnet50_2.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'

IMAGE_PATH = './bottle_000.png'
SAVE_IMAGE_PATH = './output.png'
IMAGE_RESIZE = 256
IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('SPADE', IMAGE_PATH, SAVE_IMAGE_PATH)
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
# 
# ======================

def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    dist_matrix = x[:, None] - y[None, :]
    dist_matrix = np.sum(np.power(dist_matrix, 2), axis=2)
    dist_matrix = np.sqrt(dist_matrix)

    return dist_matrix


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


def preprocess_aug(img, mask=False, angle_range=[-10, 10], return_refs=False):
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

    # for visualize
    img_resized = img.copy()

    # random rotate
    if not mask:
        h, w = img.shape[:2]
        angle = np.random.randint(angle_range[0], angle_range[0] + 1)
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(src=img,
                             M=rot_mat,
                             dsize=(w, h),
                             borderMode=cv2.BORDER_REPLICATE,
                             flags=cv2.INTER_LINEAR)

    # random crop
    if not mask:
        h, w = img.shape[:2]
        pad_h = np.random.randint(0, (h - crop_size))
        pad_w = np.random.randint(0, (w - crop_size))
        img = img[pad_h:pad_h + crop_size, pad_w:pad_w + crop_size, :]

    # normalize
    if not mask:
        img = normalize_image(img.astype(np.float32), 'ImageNet')
    else:
        img = img / 255

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    if return_refs:
        return img, img_resized, angle, pad_h, pad_w
    else:
        return img


def get_train_outputs(net):
    if args.feat:
        logger.info('loading train set feature from: %s' % args.feat)
        with open(args.feat, 'rb') as f:
            train_outputs = pickle.load(f)
        logger.info('loaded.')
        return train_outputs

    batch_size = int(args.batch_size)

    train_dir = args.train_dir
    train_imgs = sorted([
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
    ])
    if len(train_imgs) == 0:
        logger.error("train images not found in '%s'" % train_dir)
        sys.exit(-1)

    if not args.aug:
        logger.info('extract train set features without augmentation')
        aug_num = 1
    else:
        logger.info('extract train set features with augmentation')
        aug_num = args.aug_num

    train_outputs = OrderedDict([
        ('layer1', []), ('layer2', []), ('layer3', []),
        ('avgpool', []),
    ])
    for i_aug in range(aug_num):
        for i_img in range(0, len(train_imgs), batch_size):
            # prepare input data
            imgs = []
            if not args.aug:
                logger.info('from (%s ~ %s) ' %
                            (train_imgs[i_img],
                             train_imgs[min(len(train_imgs) - 1,
                                            i_img + batch_size)]))
            else:
                logger.info('from (%s ~ %s) on augmentation lap %d' %
                            (train_imgs[i_img],
                             train_imgs[min(len(train_imgs) - 1,
                                            i_img + batch_size)], i_aug))

            for image_path in train_imgs[i_img:i_img + batch_size]:
                img = load_image(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

                if not args.aug:
                    img = preprocess(img)
                else:
                    img = preprocess_aug(img)

                imgs.append(img)

            imgs = np.vstack(imgs)

            # inference
            _ = net.predict(imgs)

            feat_names = ("356", "398", "460", "493")
            for key, name in zip(train_outputs.keys(), feat_names):
                x = net.get_blob_data(name)
                train_outputs[key].append(x)

    for k, v in train_outputs.items():
        train_outputs[k] = np.vstack(v)

    # save learned distribution
    train_feat_file = "%s.pkl" % os.path.basename(train_dir)
    logger.info('saving train set feature to: %s ...' % train_feat_file)
    with open(train_feat_file, 'wb') as f:
        pickle.dump(train_outputs, f)
    logger.info('saved.')

    return train_outputs


def recognize_from_image(net):
    batch_size = int(args.batch_size)

    train_outputs = get_train_outputs(net)

    gt_type_dir = args.gt_dir if args.gt_dir else None
    test_imgs = []
    gt_imgs = []
    angle_list = []
    pad_h_list = []
    pad_w_list = []

    if not args.aug:
        logger.info('infer without augmentation')
        aug_num = 1
    else:
        logger.info('infer with augmentation')
        aug_num = args.aug_num
    N = 0
    dist_list = []
    if not args.aug:
        aug_num = 1
    else:
        aug_num = args.aug_num
    for i_aug in range(aug_num):
        test_outputs = OrderedDict([
            ('layer1', []), ('layer2', []), ('layer3', []),
            ('avgpool', []),
        ])
        for i_img in range(0, len(args.input), batch_size):
            # prepare input data
            imgs = []
            if not args.aug:
                logger.info('from (%s ~ %s) ' %
                            (args.input[i_img],
                             args.input[min(len(args.input) - 1,
                                            i_img + batch_size)]))
            else:
                logger.info('from (%s ~ %s) on augmentation lap %d' %
                            (args.input[i_img],
                             args.input[min(len(args.input) - 1,
                                            i_img + batch_size)], i_aug))
            for image_path in args.input[i_img:i_img + batch_size]:
                img = load_image(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                if not args.aug:
                    img = preprocess(img)
                    test_imgs.append(img[0])
                else:
                    (img, img_resized, angle,
                     pad_h, pad_w) = preprocess_aug(img, return_refs=True)
                    test_imgs.append(img_resized)
                    angle_list.append(angle)
                    pad_h_list.append(pad_h)
                    pad_w_list.append(pad_w)
                imgs.append(img)

                # ground truth
                gt_img = None
                if gt_type_dir:
                    fname = os.path.splitext(os.path.basename(image_path))[0]
                    gt_fpath = os.path.join(gt_type_dir, fname + '_mask.png')
                    if os.path.exists(gt_fpath):
                        gt_img = load_image(gt_fpath)
                        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2RGB)
                        if not args.aug:
                            gt_img = preprocess(gt_img, mask=True)
                            if gt_img is not None:
                                gt_img = gt_img[0, [0]]
                            else:
                                gt_img = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE))
                        else:
                            gt_img = preprocess_aug(gt_img, mask=True)
                            if gt_img is not None:
                                gt_img = gt_img[0, [0]]
                            else:
                                gt_img = np.zeros((1, IMAGE_RESIZE, IMAGE_RESIZE))

                gt_imgs.append(gt_img)

            # countup N
            N += len(imgs)

            imgs = np.vstack(imgs)

            logger.debug(f'input images shape: {imgs.shape}')
            # net.set_input_shape(imgs.shape)

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

            feat_names = ("356", "398", "460", "493")
            for key, name in zip(test_outputs.keys(), feat_names):
                x = net.get_blob_data(name)
                test_outputs[key].append(x)

        for k, v in test_outputs.items():
            test_outputs[k] = np.vstack(v)

        dist_matrix = calc_dist_matrix(
            test_outputs['avgpool'].reshape(test_outputs['avgpool'].shape[0], -1),
            train_outputs['avgpool'].reshape(train_outputs['avgpool'].shape[0], -1))

        top_k = 5
        topk_indexes = np.argsort(dist_matrix, axis=1)
        topk_indexes = topk_indexes[:, :top_k]
        print(topk_indexes)
        print(topk_indexes.shape)
        1 / 0

        print("---", test_outputs['avgpool'].shape)
        for t_idx in range(test_outputs['avgpool'].shape[0]):
            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer
                score_map = np.ones((1, 1, 224, 224))
                score_maps.append(score_map)

            # score_map = torch.mean(torch.cat(score_maps, 0), dim=0)
            dist_list.append(score_map)

    dist_list = np.vstack(dist_list)
    # dist_list = dist_list.reshape(N, H, W)
    print(dist_list.shape)
    1 / 0

    if not args.aug:
        # upsample
        score_map = np.asarray([
            np.array(Image.fromarray(s).resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
            ) for s in dist_list
        ])
    else:
        # upsample and reverse augmentation
        score_map = np.zeros([N, IMAGE_RESIZE, IMAGE_RESIZE])
        for i in range(score_map.shape[0]):
            score_map_tmp = dist_list[i]
            score_map_tmp = Image.fromarray(score_map_tmp)
            score_map_tmp = score_map_tmp.resize((IMAGE_SIZE, IMAGE_SIZE),
                                                 resample=Image.BILINEAR)
            score_map_tmp = np.array(score_map_tmp)
            # reverse crop
            pad_top = pad_h_list[i]
            pad_left = pad_w_list[i]
            pad_bottom = IMAGE_RESIZE - IMAGE_SIZE - pad_h
            pad_right = IMAGE_RESIZE - IMAGE_SIZE - pad_w
            score_map_tmp = np.pad(score_map_tmp, ((pad_top, pad_bottom),
                                                   (pad_left, pad_right)))
            # reverse rotate
            angle = angle_list[i]
            rot_mat = cv2.getRotationMatrix2D((IMAGE_RESIZE / 2, IMAGE_RESIZE / 2), -angle, 1)
            score_map_tmp = cv2.warpAffine(src=score_map_tmp,
                                           M=rot_mat,
                                           dsize=(IMAGE_RESIZE, IMAGE_RESIZE),
                                           borderMode=cv2.BORDER_REPLICATE,
                                           flags=cv2.INTER_LINEAR)
            score_map[i] = score_map_tmp
        score_map = score_map.reshape(args.aug_num, -1, IMAGE_RESIZE, IMAGE_RESIZE)
        score_map = np.mean(score_map, axis=0)

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

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = args.env_id

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
