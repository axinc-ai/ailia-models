import numpy as np
import os
import cv2
from collections import OrderedDict
import random
import pickle

from PIL import Image
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402

from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

from sklearn.metrics import precision_recall_curve

from skimage import morphology
from skimage.segmentation import mark_boundaries

WEIGHT_RESNET18_PATH = 'resnet18.onnx'
MODEL_RESNET18_PATH = 'resnet18.onnx.prototxt'
WEIGHT_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx'
MODEL_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx.prototxt'

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


def preprocess(img, size, crop_size, mask=False, keep_aspect = True):
    h, w = img.shape[:2]

    # resize
    if keep_aspect:
        if h > w:
            size = (size, int(size * h / w))
        else:
            size = (int(size * w / h), size)
    else:
        size = (size, size)
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


def preprocess_aug(img, size, crop_size, mask=False, keep_aspect = True, angle_range=[-10, 10], return_refs=False):
    h, w = img.shape[:2]

    # resize
    if keep_aspect:
        if h > w:
            size = (size, int(size * h / w))
        else:
            size = (int(size * w / h), size)
    else:
        size = (size, size)
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


def capture_training_frames_from_video(train_dir):
    if os.path.isfile(train_dir):
        capture = cv2.VideoCapture(train_dir)
    else:
        capture = cv2.VideoCapture(int(train_dir))
    if not capture:
        logger.error("file open failed")
        sts.exit(-1)
    train_imgs = []
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        cv2.imshow("capture", frame)
        train_imgs.append(frame)
        if len(train_imgs) >= 200:
            break
    capture.release()
    cv2.destroyAllWindows()
    return train_imgs


def training(net, params, size, crop_size, keep_aspect, batch_size, train_dir, aug, aug_num, seed, logger):
    # set seed
    random.seed(seed)
    idx = random.sample(range(0, params["t_d"]), params["d"])

    if os.path.isdir(train_dir):
        train_imgs = sorted([
            os.path.join(train_dir, f) for f in os.listdir(train_dir)
            if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
        ])
        if len(train_imgs) == 0:
            logger.error("train images not found in '%s'" % train_dir)
            sys.exit(-1)
    else:
        logger.info("capture 200 frames from video")
        train_imgs = capture_training_frames_from_video(train_dir)

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
                            (i_img,
                             min(len(train_imgs) - 1,
                                            i_img + batch_size)))
            else:
                logger.info('from (%s ~ %s) on augmentation lap %d' %
                            (i_img,
                             min(len(train_imgs) - 1,
                                            i_img + batch_size), i_aug))
            for image_path in train_imgs[i_img:i_img + batch_size]:
                if type(image_path) is str:
                    img = load_image(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
                if not aug:
                    img = preprocess(img, size, crop_size, keep_aspect=keep_aspect)
                else:
                    img = preprocess_aug(img, size, crop_size, keep_aspect=keep_aspect)
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

    cov_inv = np.zeros(cov.shape)
    for i in range(H * W):
        cov_inv[:, :, i] = np.linalg.inv(cov[:, :, i])
    
    train_outputs = [mean, cov, cov_inv, idx]
    return train_outputs

def infer(net, params, train_outputs, img, crop_size):
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
    idx = train_outputs[3]
    embedding_vectors = embedding_vectors[:, idx, :, :]

    # reshape 2d pixels to 1d features
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(B, C, H * W)

    # calculate distance matrix
    dist_tmp = np.zeros([B, (H * W)])
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = train_outputs[2][:, :, i] # calculate inverse on training phase (np.linalg.inv(train_outputs[1][:, :, i]))
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_tmp[:, i] = dist

    # upsample
    dist_tmp = dist_tmp.reshape(H, W)
    dist_tmp = np.array(Image.fromarray(dist_tmp).resize(
            (crop_size, crop_size), resample=Image.BILINEAR)
        )

    # apply gaussian smoothing on the score map
    dist_tmp = gaussian_filter(dist_tmp, sigma=4)

    return dist_tmp


def normalize_scores(score_map, crop_size, roi_img = None):
    N = len(score_map)
    score_map = np.vstack(score_map)
    score_map = score_map.reshape(N, crop_size, crop_size)

    if not(roi_img is None):
        roi_img [roi_img <= 0.5] = 0.0
        roi_img [roi_img > 0.5] = 1.0
        for i in range(N):
            score_map[i, :, :] *= roi_img[0, 0, :, :]

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    return scores

def calculate_anormal_scores(score_map, crop_size):
    N = len(score_map)
    score_map = np.vstack(score_map)
    score_map = score_map.reshape(N, crop_size, crop_size)

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


def get_params(arch):
    # model settings
    info = {
        "resnet18": (
            WEIGHT_RESNET18_PATH, MODEL_RESNET18_PATH,
            ("140", "156", "172"), 448, 100),
        "wide_resnet50_2": (
            WEIGHT_WIDE_RESNET50_2_PATH, MODEL_WIDE_RESNET50_2_PATH,
            ("356", "398", "460"), 1792, 550),
    }
    weight_path, model_path, feat_names, t_d, d = info[arch]

    # create param
    params = {
        "feat_names": feat_names,
        "t_d": t_d,
        "d": d,
    }

    return weight_path, model_path, params


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def visualize(img, score, threshold):
    heat_map = score * 255
    mask = score
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')  
    return heat_map, mask, vis_img


def pack_visualize(heat_map, mask, vis_img, scores, crop_size):
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    heat_map = (heat_map - scores.min() * 255) / (scores.max() * 255 - scores.min() * 255)
    heat_map = (heat_map * 255).astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    frame = np.zeros((crop_size, crop_size * 3, 3), dtype=np.uint8)
    frame[:,0:crop_size,:] = heat_map
    frame[:,crop_size:crop_size*2,:] = mask
    frame[:,crop_size*2:crop_size*3,:] = vis_img

    return frame
