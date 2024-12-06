import os
import sys
import time

import numpy as np
import cv2
import rasterio as rio
import maxflow as mf

import ailia

from detectree_utils import pixel_features

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'tile.png'
SAVE_IMAGE_PATH = 'output.png'

TREE_VAL = 255
NONTREE_VAL = 0
REFINE = True
REFINE_BETA = 50
REFINE_INT_RESCALE = 10000
MOORE_NEIGHBORHOOD_ARR = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'detectree', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

# ======================
# Parameters 2
# ======================
WEIGHT_PATH = 'detectree.onnx'
MODEL_PATH = 'detectree.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/detectree/'

# ======================
# Utils
# ======================
def load_image(input_path):
    with rio.open(input_path) as src:
        arr = src.read()
    return np.rollaxis(arr[:3], 0, 3)

def preprocess(img):
    return pixel_features.PixelFeaturesBuilder().build_features_from_arr(img)

def post_process(output, img_shape):
    if not REFINE:
        y_pred = output[0].reshape(img_shape).astype(int)*TREE_VAL
    else:
        p_nontree, p_tree = np.hsplit(output[1], 2)
        g = mf.Graph[int]()
        node_ids = g.add_grid_nodes(img_shape)
        P_nontree = p_nontree.reshape(img_shape)
        P_tree = p_tree.reshape(img_shape)

        D_tree = (REFINE_INT_RESCALE * np.log(P_nontree)).astype(int)
        D_nontree = (REFINE_INT_RESCALE * np.log(P_tree)).astype(int)
        g.add_grid_edges(node_ids, REFINE_BETA, structure=MOORE_NEIGHBORHOOD_ARR)
        g.add_grid_tedges(node_ids, D_tree, D_nontree)
        g.maxflow()
        y_pred = np.full(img_shape, NONTREE_VAL)
        y_pred[g.get_grid_segments(node_ids)] = TREE_VAL
    return y_pred

def segment_image(img, net):
    img_shape = img.shape[:2]
    img = preprocess(img)

    if args.onnx:
        input_name = net.get_inputs()[0].name
        output = net.run(None, {input_name: img})
    else:
        output = net.predict(img)

    out = post_process(output, img_shape)
    out = out.astype(np.uint8)
    return out

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                model_out = segment_image(img, net)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            model_out = segment_image(img, net)

        # save result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, model_out)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = get_capture(args.video)
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if args.savepath != SAVE_IMAGE_PATH:
        writer = get_writer(args.savepath, video_height, video_width)
    else:
        writer = None

    frame_names = None
    frame_shown = False
    frame_idx = 0
    while(True):
        ret, img = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        model_out = segment_image(img, net)
        print(type(model_out))
        print(model_out.dtype)
        print(model_out.shape)
        print(np.min(model_out), np.max(model_out))

        cv2.imshow('frame', model_out)
        if frame_names is not None:
            cv2.imwrite(f'video_{frame_idx}.png', model_out)

        if writer is not None:
            writer.write(model_out)

        frame_shown = True
        frame_idx = frame_idx + 1

    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
