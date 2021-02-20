import sys, os
import time
from logging import getLogger

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402

from trid_photo_inpainting_utils import *  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_MIDAS_PATH = 'MiDaS_model.onnx'
WEIGHT_EDGE_PATH = 'edge-model.onnx'
WEIGHT_DEPTH_PATH = 'depth-model.onnx'
WEIGHT_COLOR_PATH = 'color-model.onnx'
MODEL_MIDAS_PATH = 'MiDaS_model.onnx.prototxt'
MODEL_EDGE_PATH = 'edge-model.onnx.prototxt'
MODEL_DEPTH_PATH = 'depth-model.onnx.prototxt'
MODEL_COLOR_PATH = 'color-model.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/3d-photo-inpainting/'

IMAGE_PATH = 'moon.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    '3d-photo-inpainting model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def preprocess(img):
    img = img / 255

    height_orig = img.shape[0]
    width_orig = img.shape[1]
    unit_scale = 384.

    if width_orig > height_orig:
        scale = width_orig / unit_scale
    else:
        scale = height_orig / unit_scale

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img_resized = np.transpose(img_resized, (2, 0, 1))
    img_resized = np.expand_dims(img_resized, axis=0)

    return img_resized


def postprocess(depth, height, width):
    depth = np.squeeze(depth[0, :, :, :])
    depth = cv2.blur(depth, (3, 3))
    depth = cv2.resize(
        depth, (width, height), interpolation=cv2.INTER_AREA
    )

    depth_min = depth.min()
    depth_max = depth.max()

    bits = 2
    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    out = out.astype("uint16")
    return depth, out


def make_video(image, depth, net_info):
    config = {
        'longer_side_len': 960,
        'depth_threshold': 0.04,
        'sparse_iter': 5,
        'filter_size': [7, 7, 5, 5, 5],
        'sigma_s': 4.0,
        'sigma_r': 0.5,
        'extrapolate_border': True,
        'extrapolation_thickness': 60,
    }
    H, W = image.shape[:2]

    output_h, output_w = depth.shape
    frac = config['longer_side_len'] / max(output_h, output_w)
    output_h, output_w = int(output_h * frac), int(output_w * frac)
    image = cv2.resize(image, (output_w, output_h), interpolation=cv2.INTER_AREA)

    depth = read_MiDaS_depth(depth, 3.0, output_h, output_w)
    mean_loc_depth = depth[depth.shape[0] // 2, depth.shape[1] // 2]

    vis_photos, vis_depths = sparse_bilateral_filtering(
        depth.copy(), image.copy(),
        config, num_iter=config['sparse_iter'])
    depth = vis_depths[-1]

    int_mtx = np.array(
        [[max(H, W), 0, W // 2], [0, max(H, W), H // 2], [0, 0, 1]]
    ).astype(np.float32)
    if int_mtx.max() > 1:
        int_mtx[0, :] = int_mtx[0, :] / float(W)
        int_mtx[1, :] = int_mtx[1, :] / float(H)

    logger.info(f"Writing depth ply (and basically doing everything) at {time.time()}")
    rt_info = write_ply(
        image, depth,
        int_mtx, config,
        net_info['color'],
        net_info['edge'],
        net_info['edge'],
        net_info['depth']
    )

    verts, colors, faces, Height, Width, hFov, vFov = rt_info


# ======================
# Main functions
# ======================


def predict(img, net_info):
    h, w = img.shape[:2]
    scale = 640. / max(h, w)
    target_height, target_width = int(round(h * scale)), int(round(w * scale))

    img = preprocess(img)

    net = net_info["MiDaS"]
    if not args.onnx:
        net.set_input_shape(img.shape)
        output = net.predict({'img': img})
    else:
        output = net.run(
            ['out'],
            {'img': img})
    output = output[0]

    depth, out = postprocess(output, target_height, target_width)

    return depth, out


def recognize_from_image(image_path, net_info):
    # prepare input data
    img = load_image(image_path)
    logger.debug(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        # if not args.onnx:
        #    net.set_profile_mode(True)
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pred = predict(img, net_info)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
        # if not args.onnx:
        #    print(net.get_summary())
    else:
        depth, out = predict(img, net_info)

    # plot result
    savepath = get_savepath(args.savepath, image_path)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, out)

    make_video(img, depth, net_info)

    logger.info('Script finished successfully.')


def recognize_from_video(video, net):
    capture = get_capture(video)

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        depth = predict(frame, net)

        # plot result
        cv2.imshow('frame', depth)

    capture.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('=== MiDaS model ===')
    check_and_download_models(WEIGHT_MIDAS_PATH, MODEL_MIDAS_PATH, REMOTE_PATH)
    logger.info('=== edge model ===')
    check_and_download_models(WEIGHT_EDGE_PATH, MODEL_EDGE_PATH, REMOTE_PATH)
    logger.info('=== depth model ===')
    check_and_download_models(WEIGHT_DEPTH_PATH, MODEL_DEPTH_PATH, REMOTE_PATH)
    logger.info('=== color model ===')
    check_and_download_models(WEIGHT_COLOR_PATH, MODEL_COLOR_PATH, REMOTE_PATH)

    # initialize
    if not args.onnx:
        env_id = args.env_id
        net_midas = ailia.Net(MODEL_MIDAS_PATH, WEIGHT_MIDAS_PATH, env_id=env_id)
        net_edge = ailia.Net(MODEL_EDGE_PATH, WEIGHT_EDGE_PATH, env_id=env_id)
        net_depth = ailia.Net(MODEL_DEPTH_PATH, WEIGHT_DEPTH_PATH, env_id=env_id)
        net_color = ailia.Net(MODEL_COLOR_PATH, WEIGHT_COLOR_PATH, env_id=env_id)
    else:
        import onnxruntime
        net_midas = onnxruntime.InferenceSession(WEIGHT_MIDAS_PATH)
        net_edge = onnxruntime.InferenceSession(WEIGHT_EDGE_PATH)
        net_depth = onnxruntime.InferenceSession(WEIGHT_DEPTH_PATH)
        net_color = onnxruntime.InferenceSession(WEIGHT_COLOR_PATH)

    net_info = {
        "MiDaS": net_midas,
        "edge": net_edge,
        "depth": net_depth,
        "color": net_color,
    }

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, net_info)
    else:
        # image mode
        for image_path in args.input:
            logger.info(image_path)
            recognize_from_image(image_path, net_info)


if __name__ == '__main__':
    main()
