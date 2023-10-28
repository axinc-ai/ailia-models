from tqdm import tqdm
import logging
import os
import pickle
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Tuple

import ailia
import cv2
import faiss
import numpy as np
from detector_utils import load_image  # noqa: E402
from image_utils import normalize_image  # noqa: E402
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.metrics import precision_recall_curve
from sklearn.random_projection import SparseRandomProjection

from k_center_greedy import KCenterGreedy

IMAGE_SIZE = 224

WEIGHT_RESNET18_PATH = "resnet18.onnx"
MODEL_RESNET18_PATH = "resnet18.onnx.prototxt"
WEIGHT_WIDE_RESNET50_2_PATH = "wide_resnet50_2.onnx"
MODEL_WIDE_RESNET50_2_PATH = "wide_resnet50_2.onnx.prototxt"
MODEL_SETTINGS = {
    "resnet18": (
        WEIGHT_RESNET18_PATH,
        MODEL_RESNET18_PATH,
        ("140", "156", "172"),
        448,
        100,
    ),
    "wide_resnet50_2": (
        WEIGHT_WIDE_RESNET50_2_PATH,
        MODEL_WIDE_RESNET50_2_PATH,
        ("356", "398", "460"),
        1792,
        550,
    ),
}


def vstack(array: List[np.ndarray]) -> np.ndarray:
    return np.vstack(array)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def calculate_anormal_scores(score_map: List[np.ndarray]) -> np.ndarray:
    """
    Calculate anomal scores for each samples.
    Calculated anomal scores are the maximum value of each score map.

    Args:
        score_map List[np.ndarray]: list of score map for each samples

    Returns:
        np.ndarray: anomal scores for each samples
    """
    length = len(score_map)
    score_map = vstack(score_map)
    score_map = score_map.reshape(length, IMAGE_SIZE, IMAGE_SIZE)

    # Calculated anormal score
    anormal_scores = np.zeros((score_map.shape[0]))
    for i in range(score_map.shape[0]):
        anormal_scores[i] = score_map[i].max()

    return anormal_scores


def normalize_score_maps(score_map: List[np.ndarray]) -> np.ndarray:
    """
    Normalize score maps for each samples

    Args:
        score_map List[np.ndarray]: list of score map for each samples

    Returns:
        np.ndarray: normalized score maps
    """
    length = len(score_map)
    score_map = vstack(score_map)
    score_map = score_map.reshape(length, IMAGE_SIZE, IMAGE_SIZE)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    return scores


def visualize(
    img: np.ndarray, score: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat_map = score * 255
    mask = score
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode="thick")
    return heat_map, mask, vis_img


def embedding_concat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Concatenate feature maps

    Args:
        x np.ndarray: feature maps
        y np.ndarray: feature maps

    Returns:
        np.ndarray: concatenated feature maps
    """
    B, C1, H1, W1 = x.shape
    _, C2, H2, W2 = y.shape

    if not H1 == W1:
        raise Exception("Invalid shape")

    s = H1 // H2
    sel = [np.array([i for i in range(i, H1, s)]) for i in range(s)]

    a = np.zeros((B, C1 * s * s, H2 * W2))
    for b, c, i in product(range(B), range(C1), range(s * s)):
        a[b, c * s * s + i, :] = x[b, c, sel[i // s][:, None], sel[i % s]].flatten()

    x = a.reshape((B, C1, -1, H2, W2))
    z = np.zeros((B, C1 + C2, s * s, H2, W2))
    for i in range(s * s):
        z[:, :, i, :, :] = np.concatenate((x[:, :, i, :, :], y), axis=1)
    z = z.reshape((B, -1, H2 * W2))

    _, C3, _ = z.shape
    a = np.zeros((B, C3 // (s * s), H1, W1))
    for b, c, i in product(range(B), range(C3 // (s * s)), range(s * s)):
        x = z[b, c * s * s + i, :].reshape((H2, W2))
        a[b, c, sel[i // s][:, None], sel[i % s]] = x

    return a


def preprocess(
    img: np.ndarray, size: int, mask: bool = False, keep_aspect: bool = True
) -> np.ndarray:
    """
    Preprocess image

    Args:
        img np.ndarray: input image
        size int: resized image size
        mask bool: whether input image is for mask or not (fefault: False)
        keep_aspect bool: whether keep image aspect or not (default: True)

    Returns:
        np.ndarray: preprocessed image
    """
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

    img = np.array(
        Image.fromarray(img).resize(
            size, resample=Image.ANTIALIAS if not mask else Image.NEAREST
        )
    )

    # center crop
    h, w = img.shape[:2]
    pad_h = (h - crop_size) // 2
    pad_w = (w - crop_size) // 2
    img = img[pad_h : pad_h + crop_size, pad_w : pad_w + crop_size, :]

    # normalize
    if not mask:
        img = normalize_image(img.astype(np.float32), "ImageNet")
    else:
        img = img / 255

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def concat_embedding_layers(outputs: Dict[str, Any]) -> np.ndarray:
    return embedding_concat(outputs["layer2"], outputs["layer3"])


def generate_dataset_from_video(
    train_dir: str,
    training_image_num_upper: int = 200
) -> List[np.ndarray]:
    """
    Generate training image from video

    Args:
        train_dir str: directory which contains video for training
        training_image_num_upper int: the maximum number of images for traininig (default: 200)
    """
    if os.path.isfile(train_dir):
        capture = cv2.VideoCapture(train_dir)
    else:
        capture = cv2.VideoCapture(int(train_dir))
    if not capture:
        logger.error("file open failed")
        sts.exit(-1)

    train_imgs = []
    while True:
        ret, frame = capture.read()

        if not ret:
            break

        train_imgs.append(frame)
        if len(train_imgs) >= training_image_num_upper:
            break

    capture.release()

    return train_imgs


def reshape_embedding(embedding: np.ndarray) -> List[np.ndarray]:
    embedding_list = []
    B, _, H, W = embedding.shape

    for b, h, w in tqdm(product(range(B), range(H), range(W)), total=B*H*W):
        embedding_list.append(embedding[b, :, h, w])

    return embedding_list


def training(
    net: ailia.wrapper.Net,
    params: Dict[str, Any],
    size: int,
    keep_aspect: bool,
    batch_size: int,
    train_dir: str,
    aug: bool,
    aug_num: int,
    coreset_sampling_ratio: float,
    logger: logging.Logger,
) -> np.ndarray:
    if os.path.isdir(train_dir):
        train_imgs = sorted(
            [
                os.path.join(train_dir, f)
                for f in os.listdir(train_dir)
                if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")
            ]
        )
        if len(train_imgs) == 0:
            logger.error("train images not found in '%s'" % train_dir)
            sys.exit(-1)
    else:
        logger.info("capture 200 frames from video")
        train_imgs = generate_dataset_from_video(train_dir)

    if not aug:
        logger.info("extract train set features without augmentation")
        aug_num = 1
    else:
        logger.info("extract train set features with augmentation")
        aug_num = aug_num

    embedding_vectors: Optional[np.ndarray] = None

    for i_aug in range(aug_num):
        for i_img in range(0, len(train_imgs), batch_size):
            imgs = []

            if not aug:
                logger.info(
                    "from (%s ~ %s) "
                    % (i_img, min(len(train_imgs) - 1, i_img + batch_size))
                )
            else:
                logger.info(
                    "from (%s ~ %s) on augmentation lap %d"
                    % (i_img, min(len(train_imgs) - 1, i_img + batch_size), i_aug)
                )
            for image_path in train_imgs[i_img : i_img + batch_size]:
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

            imgs = vstack(imgs)
            logger.debug(f"input image shape: {imgs.shape}")
            net.set_input_shape(imgs.shape)

            # inference
            _ = net.predict(imgs)
            train_outputs = OrderedDict([("layer2", []), ("layer3", [])])
            for key, name in zip(train_outputs.keys(), params["feat_names"]):
                train_outputs[key].append(net.get_blob_data(name))
            for k, v in train_outputs.items():
                train_outputs[k] = v[0]

            _embedding_vectors = concat_embedding_layers(train_outputs)

            if embedding_vectors is None:
                embedding_vectors = _embedding_vectors
            else:
                embedding_vectors = np.concatenate(
                    (embedding_vectors, _embedding_vectors), axis=0
                )

    embedding_vectors = np.array(reshape_embedding(embedding_vectors))
    randomprojector = SparseRandomProjection(
        n_components="auto", eps=0.9
    )  # 'auto' => Johnson-Lindenstrauss lemma
    randomprojector.fit(embedding_vectors)

    sampler = KCenterGreedy(embedding_vectors, 0, 0)
    selected_idx = sampler.select_batch(
        model=randomprojector,
        already_selected=[],
        N=int(embedding_vectors.shape[0] * coreset_sampling_ratio),
    )
    embedding_coreset = embedding_vectors[selected_idx]
    print("initial embedding size : ", embedding_vectors.shape)  #  (245760, 1536)
    print("final embedding size : ", embedding_coreset.shape)

    index = faiss.IndexFlatL2(embedding_coreset.shape[1])
    index.add(embedding_coreset)
    faiss.write_index(index, "embeddings/index.faiss")
    return embedding_coreset


def get_params(arch: str, n_neighbors: int) -> Tuple[str, str, Dict[str, Any]]:
    weight_path, model_path, feat_names, t_d, d = MODEL_SETTINGS[arch]
    params = {
        "feat_names": feat_names,
        "t_d": t_d,
        "d": d,
        "n_neighbors": n_neighbors,
    }
    return weight_path, model_path, params


def min_max_norm(image: np.ndarray) -> np.ndarray:
    """
    Normalize image by maximum and minimum value

    Args:
        image np.ndarray: target iamge

    Returns:
        np.ndarray: normalized image
    """
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def heatmap_on_image(heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Synthesize heatmap and original image

    Args:
        heatmap np.ndarray: heatmap 
        image np.ndarray: original image

    Returns:
        np.ndarray synthesized image
    """
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def cvt2heatmap(gray: np.ndarray) -> np.ndarray:
    """
    Convert gray scale image to color map

    Args:
        gray np.ndarray: gray scale image

    Returns:
        np.ndarray: color map image
    """
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def infer(
    net: ailia.wrapper.Net,
    params: Dict[str, Any],
    embedding_coreset: np.ndarray,
    img: np.ndarray,
):
    height, width = img.shape[-2:]

    _ = net.predict([img])
    test_outputs = OrderedDict([("layer2", []), ("layer3", [])])
    for key, name in zip(test_outputs.keys(), params["feat_names"]):
        test_outputs[key].append(net.get_blob_data(name))
    for k, v in test_outputs.items():
        test_outputs[k] = v[0]

    embedding_vectors = concat_embedding_layers(test_outputs)

    index = faiss.read_index("embeddings/index.faiss")
    embedding_test = np.array(reshape_embedding(embedding_vectors))
    score_patches, _ = index.search(embedding_test, k=params["n_neighbors"])

    anomaly_map = score_patches[:, 0].reshape((56, 56))
    N_b = score_patches[np.argmax(score_patches[:, 0])]
    w = 1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b)))
    score = w * max(score_patches[:, 0])  # Image-level score

    anomaly_map_resized = cv2.resize(anomaly_map, (width, height))
    anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
    pred_px_lvl = anomaly_map_resized_blur.ravel()
    pred_img_lvl = score

    save_anomaly_map(anomaly_map_resized_blur, img[0].transpose(1, 2, 0))
    return anomaly_map_resized_blur


def save_anomaly_map(anomaly_map: np.ndarray, input_img: np.ndarray):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    anomaly_map_norm = min_max_norm(anomaly_map)
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap, input_img)
    cv2.imwrite("heatmap.png", hm_on_img)
