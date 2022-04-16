import sys
import os
import time
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
from nms_utils import nms_boxes  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'd4lcn.onnx'
MODEL_PATH = 'd4lcn.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/d4lcn/'

IMAGE_PATH = '000001.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'D4LCN', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--depth_path', type=str, default=None,
    help='the label file (object labels for image) or stored directory path.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def get_depth(path, name):
    if path is None:
        file_path = "depth/%s.png" % name
        if os.path.exists(file_path):
            logger.info("depth file: %s" % file_path)
            return file_path
        else:
            return None
    elif os.path.isdir(path):
        file_path = "%s/%s.png" % (path, name)
        if os.path.exists(file_path):
            logger.info("depth file: %s" % file_path)
            return file_path
    elif os.path.exists(path):
        logger.info("depth file: %s" % path)
        return path

    logger.error("depth file is not found. (path: %s)" % path)
    sys.exit(-1)


def bbox_transform_inv(boxes, deltas, means=None, stds=None):
    """
    Compute the bbox target transforms in 3D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if stds is not None:
        dx *= stds[0]
        dy *= stds[1]
        dw *= stds[2]
        dh *= stds[3]

    if means is not None:
        dx += means[0]
        dy += means[1]
        dw += means[2]
        dh += means[3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros(deltas.shape)

    # x1, y1, x2, y2
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


# ======================
# Main functions
# ======================


def preprocess(img, depth):
    h, w = img.shape[:2]

    scale_factor = IMAGE_HEIGHT / h
    h = np.round(h * scale_factor).astype(int)
    w = np.round(w * scale_factor).astype(int)

    # resize
    img = cv2.resize(img.astype(np.float32), (w, h))
    depth = cv2.resize(depth.astype(np.float32), (w, h))

    # Normalize
    mean = np.array([102.9801, 115.9465, 122.7717])
    std = np.array([1., 1., 1.])
    depth_mean = np.array([4413.1606, 4413.1606, 4413.1606])
    depth_std = np.array([3270.0159, 3270.0159, 3270.0159])
    img = (img - mean) / std
    depth = (depth - depth_mean) / depth_std

    img = np.transpose(img, [2, 0, 1])
    depth = np.transpose(depth, [2, 0, 1])

    img = np.expand_dims(img, axis=0)
    depth = np.expand_dims(depth, axis=0)

    return img, depth


def locate_anchors(anchors, feat_size, stride):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.

    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across [feat_h, feat_w]
        stride (int): stride of a network

    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
    """

    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]): anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    # important to unroll according to pytorch
    shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1])
    shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1])
    shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1])
    shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1])
    anchor_tracker = anchor_tracker.reshape(1, stack_size, feat_size[1])

    shift_x1 = shift_x1.transpose(1, 2, 0).reshape(-1, 1)
    shift_y1 = shift_y1.transpose(1, 2, 0).reshape(-1, 1)
    shift_x2 = shift_x2.transpose(1, 2, 0).reshape(-1, 1)
    shift_y2 = shift_y2.transpose(1, 2, 0).reshape(-1, 1)
    anchor_tracker = anchor_tracker.transpose(1, 2, 0).reshape(-1, 1)

    rois = np.concatenate(
        (shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker),
        axis=1)

    locate_anchors.feat_h = feat_size[0]
    locate_anchors.feat_w = feat_size[1]
    locate_anchors.rois = rois

    return rois


locate_anchors.anchors = None
locate_anchors.feat_h = None
locate_anchors.feat_w = None


def post_process(anchors, bbox_2d, bbox_3d, rois, prob, scale_factor):
    bbox_x = bbox_2d[:, :, 0]
    bbox_y = bbox_2d[:, :, 1]
    bbox_w = bbox_2d[:, :, 2]
    bbox_h = bbox_2d[:, :, 3]

    bbox_x3d = bbox_3d[:, :, 0]
    bbox_y3d = bbox_3d[:, :, 1]
    bbox_z3d = bbox_3d[:, :, 2]
    bbox_w3d = bbox_3d[:, :, 3]
    bbox_h3d = bbox_3d[:, :, 4]
    bbox_l3d = bbox_3d[:, :, 5]
    bbox_ry3d = bbox_3d[:, :, 6]

    bbox_means = np.array(
        [[-0.00022546, 0.00160404, 0.06383215, -0.09315256, 0.01069604, -0.06744095,
          0.19155604, 0.05884239, -0.02122913, 0.06871941, -0.00352113]])
    bbox_stds = np.array(
        [[0.13962965, 0.1255247, 0.24738377, 0.23853353, 0.16330168, 0.13235298,
          3.62072376, 0.38246312, 0.10154974, 0.50257567, 1.85493732]])

    # detransform 3d
    bbox_x3d = bbox_x3d * bbox_stds[:, 4][0] + bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * bbox_stds[:, 5][0] + bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * bbox_stds[:, 6][0] + bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * bbox_stds[:, 7][0] + bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * bbox_stds[:, 8][0] + bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * bbox_stds[:, 9][0] + bbox_means[:, 9][0]
    bbox_ry3d = bbox_ry3d * bbox_stds[:, 10][0] + bbox_means[:, 10][0]

    # find 3d source
    tracker = rois[:, 4].astype(np.int64)
    src_3d = anchors[tracker, 4:]

    # compute 3d transform
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d = bbox_x3d[0, :] * widths + ctr_x
    bbox_y3d = bbox_y3d[0, :] * heights + ctr_y
    bbox_z3d = src_3d[:, 0] + bbox_z3d[0, :]
    bbox_w3d = np.exp(bbox_w3d[0, :]) * src_3d[:, 1]
    bbox_h3d = np.exp(bbox_h3d[0, :]) * src_3d[:, 2]
    bbox_l3d = np.exp(bbox_l3d[0, :]) * src_3d[:, 3]
    bbox_ry3d = src_3d[:, 4] + bbox_ry3d[0, :]

    # bundle
    coords_3d = np.stack(
        (bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]], bbox_w3d[:bbox_x3d.shape[0]],
         bbox_h3d[:bbox_x3d.shape[0]], bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]),
        axis=1)

    # compile deltas pred
    deltas_2d = np.concatenate(
        (bbox_x[0, :, np.newaxis], bbox_y[0, :, np.newaxis],
         bbox_w[0, :, np.newaxis], bbox_h[0, :, np.newaxis]),
        axis=1)
    coords_2d = bbox_transform_inv(rois, deltas_2d, means=bbox_means[0, :], stds=bbox_stds[0, :])

    # scale coords
    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    prob = prob[0, :, :]
    cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
    scores = np.amax(prob[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]
    tracker = tracker[sorted_inds]

    # pre-nms
    nms_topN = 3000
    cls_pred = cls_pred[0:min(nms_topN, cls_pred.shape[0])]
    tracker = tracker[0:min(nms_topN, tracker.shape[0])]
    aboxes = aboxes[0:min(nms_topN, aboxes.shape[0]), :]
    coords_3d = coords_3d[0:min(nms_topN, coords_3d.shape[0])]

    # nms
    nms_thres = 0.4
    keep_inds = nms_boxes(aboxes[:, 0:4], aboxes[:, 4], nms_thres)

    # stack cls prediction
    aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d, tracker[:, np.newaxis]))

    # suppress boxes
    aboxes = aboxes[keep_inds, :]

    return aboxes


def predict(net, img, depth):
    h, w = img.shape[:2]
    img, depth = preprocess(img, depth)

    output = net.predict([img, depth])

    cls, prob, bbox_2d, bbox_3d = output

    anchors = locate_anchors.anchors
    feat_size = (32, 106)
    feat_stride = 16
    rois = locate_anchors(anchors, feat_size, feat_stride)

    scale_factor = img.shape[-2] / h
    aboxes = post_process(anchors, bbox_2d, bbox_3d, rois, prob, scale_factor)

    return aboxes


def recognize_from_image(net):
    depth_path = args.depth_path

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # depth
        name = os.path.splitext(os.path.basename(image_path))[0]
        depth_path = get_depth(depth_path, name)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth[:, :, np.newaxis]
        depth = np.tile(depth, (1, 1, 3))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(net, img, depth)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net, img, depth)

        save_path = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {save_path}')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # anchor
    locate_anchors.anchors = np.load('anchor.npy')

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
