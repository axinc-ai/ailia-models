import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import imread, normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'VGG19.onnx'
MODEL_PATH = 'VGG19.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dfm/'

IMAGE_A_PATH = 'img_A.png'
IMAGE_B_PATH = 'img_B.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'DFM', IMAGE_A_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-i2', '--input2', metavar='IMAGE2', default=IMAGE_B_PATH,
    help='Pair image path of input image.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def dense_feature_matching(
        map_A, map_B, ratio_th, bidirectional=True):
    # normalize and reshape feature maps
    _, ch, h_A, w_A = map_A.shape
    _, _, h_B, w_B = map_B.shape

    d1 = map_A.reshape(ch, -1).T
    d1 /= np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1)

    d2 = map_B.reshape(ch, -1).T
    d2 /= np.expand_dims(np.sqrt(np.sum(np.square(d2), axis=1)), axis=1)

    # perform matching
    matches, scores = mnn_ratio_matcher(d1, d2, ratio_th, bidirectional)

    # form a coordinate grid and convert matching indexes to image coordinates
    x_A, y_A, = np.meshgrid(np.arange(w_A), np.arange(h_A))
    x_B, y_B, = np.meshgrid(np.arange(w_B), np.arange(h_B))

    points_A = np.stack((x_A.flatten()[matches[:, 0]], y_A.flatten()[matches[:, 0]]))
    points_B = np.stack((x_B.flatten()[matches[:, 1]], y_B.flatten()[matches[:, 1]]))

    # discard the point on image boundaries
    discard = (points_A[0, :] == 0) | (points_A[0, :] == w_A - 1) | (points_A[1, :] == 0) | (points_A[1, :] == h_A - 1) \
              | (points_B[0, :] == 0) | (points_B[0, :] == w_B - 1) | (points_B[1, :] == 0) | (
                      points_B[1, :] == h_B - 1)

    points_A = points_A[:, ~discard]
    points_B = points_B[:, ~discard]

    return points_A, points_B


def mnn_ratio_matcher(
        descriptors1, descriptors2, ratio=0.8, bidirectional=True):
    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    sim = descriptors1 @ descriptors2.T

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim = -np.sort(-sim, axis=1)[:, :2]
    nns = np.argsort(-sim, axis=1)[:, :2]

    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]
    match_sim = nns_sim[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim = -np.sort(-sim.T, axis=1)[:, :2]
    nns = np.argsort(-sim.T, axis=1)[:, :2]
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]

    # if not bidirectional, do not use ratios from 2 to 1
    ratios21[:] *= 1 if bidirectional else 0

    # Mutual NN + symmetric ratio test.
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12]) & \
           ((ratios12 <= ratio) &
            (ratios21[nn12] <= ratio))  # discard ratios21 to get the same results with matlab
    # Final matches.
    matches = np.stack([ids1[mask], nn12[mask]], axis=-1)
    match_sim = match_sim[mask]

    return matches, match_sim


def draw_matches(img_A, img_B, keypoints0, keypoints1):
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(keypoints1[i][0], keypoints1[i][1], 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))

    matched_images = cv2.drawMatches(
        cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s,
        cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)

    return matched_images


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB

    img = normalize_image(img, normalize_type='ImageNet')

    # zero padding to make image canvas a multiple of padding_n
    padding_n = 16
    pad_right = 16 - im_w % padding_n if im_w % padding_n else 0
    pad_bottom = 16 - im_h % padding_n if im_h % padding_n else 0

    if pad_bottom or pad_right:
        pad_img = np.zeros((im_h + pad_bottom, im_w + pad_right, 3))
        pad_img[:im_h, :im_w, :] = img
        img = pad_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, None


def post_processing(output):
    return None


def predict(net, img_A, img_B):
    img_A, pad_A = preprocess(img_A)
    img_B, pad_B = preprocess(img_B)

    # feedforward
    activations_A = net.predict([img_A])
    activations_B = net.predict([img_B])

    # initiate warped image, its activations, initial&final estimate of homography
    H_init = np.eye(3, dtype=np.double)
    H = np.eye(3, dtype=np.double)

    # initiate matches
    ratio_th = np.array([0.9, 0.9, 0.9, 0.9, 0.95, 1.])
    bidirectional = True
    points_A, points_B = dense_feature_matching(
        activations_A[-1], activations_B[-1],
        ratio_th[-1], bidirectional)

    # upsample points (zero based)
    points_A = (points_A + 0.5) * 16 - 0.5
    points_B = (points_B + 0.5) * 16 - 0.5

    # estimate homography for initial warping
    src = points_B.T
    dst = points_A.T

    if points_A.shape[1] >= 4:
        H_init, _ = cv2.findHomography(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=16 * np.sqrt(2) + 1,
            maxIters=5000, confidence=0.9999)

        # opencv might return None for H, check for None
        H_init = np.eye(3, dtype=np.double) if H_init is None else H_init

    # warp image B onto image A
    img_C = cv2.warpPerspective(img_B, H_init, (img_A.shape[1], img_A.shape[0]))
    img_C, pad_C = preprocess(img_C)

    activations_C = net.predict([img_C])

    # initiate matches
    points_A, points_C = dense_feature_matching(
        activations_A[-2], activations_C[-2],
        ratio_th[-2], bidirectional)

    # for k in range(len(activations_A) - 3, -1, -1):
    #     points_A, points_C = refine_points(
    #         points_A, points_C, activations_A[k], activations_C[k],
    #         ratio_th[k], bidirectional)
    #
    # # warp points form C to B (H_init is zero-based, use zero-based points)
    # points_B = torch.from_numpy(np.linalg.inv(H_init)) @ torch.vstack(
    #     (points_C, torch.ones((1, points_C.size(1))))).double()
    # points_B = points_B[0:2, :] / points_B[2, :]
    #
    # points_A = points_A.double()
    #
    # # optional
    # in_image = torch.logical_and(
    #     points_A[0, :] < (inp_A.shape[3] - padding_A[0] - 16),
    #     points_A[1, :] < (inp_A.shape[2] - padding_A[1] - 16))
    # in_image = torch.logical_and(
    #     in_image,
    #     torch.logical_and(points_B[0, :] < (inp_B.shape[3] - padding_B[0] - 16),
    #                       points_B[1, :] < (inp_B.shape[3] - padding_B[1] - 16)))

    points_A = points_A[:, in_image]
    points_B = points_B[:, in_image]

    # estimate homography
    src = points_B.T
    dst = points_A.T

    if points_A.shape[1] >= 4:
        H, _ = cv2.findHomography(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000, confidence=0.9999)

        # opencv might return None for H, check for None
        H = np.eye(3, dtype=np.double) if H is None else H

    return H, H_init, points_A, points_B


def recognize_from_image(net):
    img_B = load_image(args.input2)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGRA2BGR)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img_A = load_image(image_path)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(net, img_A, img_B)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net, img_A, img_B)

        H, H_init, points_A, points_B = out
        keypoints0 = points_A.T
        keypoints1 = points_B.T

        res_img = draw_matches(img_A, img_B, keypoints0, keypoints1)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
