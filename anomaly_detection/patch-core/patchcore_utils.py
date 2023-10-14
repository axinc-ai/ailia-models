from itertools import product
from typing import Dict, Any, List
import numpy as np
import os
import cv2
from collections import OrderedDict
import random
import pickle

from PIL import Image
from image_utils import normalize_image  # noqa: E402
from detector_utils import load_image  # noqa: E402

from sklearn.random_projection import SparseRandomProjection

from sklearn.metrics import precision_recall_curve

from skimage import morphology
from skimage.segmentation import mark_boundaries

from k_center_greedy import KCenterGreedy

import faiss

IMAGE_SIZE = 224

WEIGHT_RESNET18_PATH = 'resnet18.onnx'
MODEL_RESNET18_PATH = 'resnet18.onnx.prototxt'
WEIGHT_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx'
MODEL_WIDE_RESNET50_2_PATH = 'wide_resnet50_2.onnx.prototxt'


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist


def embedding_concat(x, y):
    B, C1, H1, W1 = x.shape
    _, C2, H2, W2 = y.shape

    if not H1 == W1:
        raise Exception("Invalid shape")

    s = H1 // H2
    sel = [np.array([i for i in range(i, H1, s)]) for i in range(s)]

    a = np.zeros((B, C1 * s * s, H2 * W2))
    for b, c, i in product(range(B), range(C1), range(s*s)):
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
    for b, c, i in product(range(B), range(C3 // (s*s)), range(s*s)):
        x = z[b, c * s * s + i, :].reshape((H2, W2))
        a[
            b, c, sel[i // s][:, None], sel[i % s]
        ] = x

    return a


def preprocess(img, size, mask=False, keep_aspect = True):
    h, w = img.shape[:2]
    crop_size = IMAGE_SIZE

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


def postprocess(outputs: Dict[str, Any]):
    return embedding_concat(outputs["layer2"], outputs["layer3"])


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


def reshape_embedding(embedding: np.ndarray) -> List[np.ndarray]:
    embedding_list = []

    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])

    return embedding_list


def training(
    net,
    params,
    size: int,
    keep_aspect,
    batch_size: int,
    train_dir: str,
    aug,
    aug_num,
    coreset_sampling_ratio: float,
    logger
):
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


    embedding_vectors = None

    for i_aug in range(aug_num):
        for i_img in range(0, len(train_imgs), batch_size):
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
                if isinstance(image_path, str):
                    img = load_image(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
                if not aug:
                    img = preprocess(img, size, keep_aspect=keep_aspect)
                else:
                    img = preprocess_aug(img, size, keep_aspect=keep_aspect)
                imgs.append(img)

            imgs = np.vstack(imgs)
            logger.debug(f"input image shape: {imgs.shape}")
            net.set_input_shape(imgs.shape)

            # inference
            _ = net.predict(imgs)
            train_outputs = OrderedDict([
                ("layer2", []), ("layer3", [])
            ])
            for key, name in zip(train_outputs.keys(), params["feat_names"]):
                train_outputs[key].append(net.get_blob_data(name))
            for k, v in train_outputs.items():
                train_outputs[k] = v[0]

            _embedding_vectors = postprocess(train_outputs)

            if embedding_vectors is None:
                embedding_vectors = _embedding_vectors
            else:
                embedding_vectors = np.concatenate((embedding_vectors, _embedding_vectors), axis=0)

    embedding_vectors = np.array(reshape_embedding(embedding_vectors))
    randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
    randomprojector.fit(embedding_vectors)

    sampler = KCenterGreedy(embedding_vectors, 0, 0)
    selected_idx = sampler.select_batch(
        model=randomprojector,
        already_selected=[],
        N=int(embedding_vectors.shape[0]*coreset_sampling_ratio),
    )
    embedding_coreset = embedding_vectors[selected_idx]
    print('initial embedding size : ', embedding_vectors.shape) #  (245760, 1536)
    print('final embedding size : ', embedding_coreset.shape)

    # faiss
    index = faiss.IndexFlatL2(embedding_coreset.shape[1])
    index.add(embedding_coreset)
    faiss.write_index(self.index, os.path.join(self.embedding_dir_path,'index.faiss'))


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

