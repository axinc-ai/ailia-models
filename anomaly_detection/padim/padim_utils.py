import numpy as np
import os
import cv2
from collections import OrderedDict
import random
import pickle

#__all__ = [
#    'embedding_concat',
#]

from PIL import Image
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402

from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

from sklearn.metrics import precision_recall_curve

IMAGE_RESIZE = 256
IMAGE_SIZE = 224

def embedding_concat(x, y):
    B, C1, H1, W1 = x.shape
    _, C2, H2, W2 = y.shape

    assert H1 == W1

    s = H1 // H2
    sel = [np.array([i for i in range(i, H1, s)]) for i in range(s)]

    a = np.zeros((B, C1 * s * s, H2 * W2))
    for b in range(B):
        for c in range(C1):
            for i in range(s * s):
                a[b, c * s * s + i, :] = x[
                    b, c, sel[i // s][:, None], sel[i % s]
                ].flatten()
    x = a.reshape((B, C1, -1, H2, W2))
    z = np.zeros((B, C1 + C2, s * s, H2, W2))
    for i in range(s * s):
        z[:, :, i, :, :] = np.concatenate((x[:, :, i, :, :], y), axis=1)
    z = z.reshape((B, -1, H2 * W2))

    _, C3, _ = z.shape
    a = np.zeros((B, C3 // (s * s), H1, W1))
    for b in range(B):
        for c in range(C3 // (s * s)):
            for i in range(s * s):
                x = z[b, c * s * s + i, :].reshape((H2, W2))
                a[
                    b, c, sel[i // s][:, None], sel[i % s]
                ] = x

    return a

#
# def embedding_concat(x, y):
#     import torch
#     import torch.nn.functional as F
#
#     x = torch.from_numpy(x)
#     y = torch.from_numpy(y)
#
#     B, C1, H1, W1 = x.size()
#     _, C2, H2, W2 = y.size()
#     s = int(H1 / H2)
#     x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
#     x = x.view(B, C1, -1, H2, W2)
#     z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
#     for i in range(x.size(2)):
#         z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
#     z = z.view(B, -1, H2 * W2)
#     z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
#
#     z = z.numpy()
#
#     return z

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


def postprocess(outputs):
    # Embedding concat
    embedding_vectors = outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, outputs[layer_name])

    return embedding_vectors

def training(net, params, idx, batch_size, train_dir, aug, aug_num, logger):
    train_imgs = sorted([
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
    ])
    if len(train_imgs) == 0:
        logger.error("train images not found in '%s'" % train_dir)
        sys.exit(-1)

    if not aug:
        logger.info('extract train set features without augmentation')
        aug_num = 1
    else:
        logger.info('extract train set features with augmentation')
        aug_num = aug_num
    mean = None
    N = 0
    for i_aug in range(aug_num):
        for i_img in range(0, len(train_imgs), batch_size):
            # prepare input data
            imgs = []
            if not aug:
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
                if not aug:
                    img = preprocess(img)
                else:
                    img = preprocess_aug(img)
                imgs.append(img)

            # countup N
            N += len(imgs)

            imgs = np.vstack(imgs)

            logger.debug(f'input images shape: {imgs.shape}')
            net.set_input_shape(imgs.shape)

            # inference
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
            embedding_vectors = embedding_vectors[:, idx, :, :]

            # reshape 2d pixels to 1d features
            B, C, H, W = embedding_vectors.shape
            embedding_vectors = embedding_vectors.reshape(B, C, H * W)

            # initialize mean and covariance matrix
            if (mean is None):
                mean = np.zeros((C, H * W), dtype=np.float32)
                cov = np.zeros((C, C, H * W), dtype=np.float32)

            # calculate multivariate Gaussian distribution
            # (add up mean and covariance matrix)
            mean += np.sum(embedding_vectors, axis=0)
            for i in range(H * W):
                # https://github.com/numpy/numpy/blob/v1.21.0/numpy/lib/function_base.py#L2324-L2543
                m = embedding_vectors[:, :, i]
                m = m - (mean[:, [i]].T / N)
                cov[:, :, i] += m.T @ m

    # devide mean by N
    mean = mean / N
    # devide covariance by N-1, and calculate inverse
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = (cov[:, :, i] / (N - 1)) + 0.01 * I

    train_outputs = [mean, cov, idx]
    return train_outputs

def infer(net, params, train_outputs, img):
    # prepare input data
    imgs = []
    imgs.append(img)
    imgs = np.vstack(imgs)

    # inference
    net.set_input_shape(imgs.shape)
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
    idx = train_outputs[2]
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

    # upsample
    dist_tmp = dist_tmp.reshape(H, W)
    dist_tmp = np.array(Image.fromarray(dist_tmp).resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        )

    # apply gaussian smoothing on the score map
    dist_tmp = gaussian_filter(dist_tmp, sigma=4)

    return dist_tmp


def normalize_scores(score_map):
    N = len(score_map)
    score_map = np.vstack(score_map)
    score_map = score_map.reshape(N, IMAGE_SIZE, IMAGE_SIZE)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    return scores

def calculate_anormal_scores(score_map):
    N = len(score_map)
    score_map = np.vstack(score_map)
    score_map = score_map.reshape(N, IMAGE_SIZE, IMAGE_SIZE)

    # Calculated anormal score
    anormal_scores = np.zeros((score_map.shape[0]))
    for i in range(score_map.shape[0]):
        anormal_scores[i] = score_map[i].max()
    
    return anormal_scores


def decide_threshold(scores, gt_imgs):
    # get optimal threshold
    gt_mask = np.asarray(gt_imgs)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold
