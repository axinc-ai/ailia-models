import sys
import time
import os

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_2x_PATH = 'FLAVR_2x.onnx'
MODEL_2x_PATH = 'FLAVR_2x.onnx.prototxt'
WEIGHT_4x_PATH = 'FLAVR_4x.onnx'
MODEL_4x_PATH = 'FLAVR_4x.onnx.prototxt'
WEIGHT_8x_PATH = 'FLAVR_8x.onnx'
MODEL_8x_PATH = 'FLAVR_8x.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/flavr/'

IMAGE_PATH = 'vimeo_septuplet/0266'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('FLAVR model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-ip', '--interpolation', type=int, choices=(2, 4, 8), default=2,
    help='2x/4x/8x Interpolation'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def postprocess(output):
    output = output.transpose((1, 2, 0)) * 255.0
    img = output.astype(np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR

    return img


def recognize_from_image(net, n_output):
    # input image loop
    input_frames = "1357"

    # Load images
    images = [load_image(pth) for pth in args.input]
    images = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) for img in images]
    images = [preprocess(img) for img in images]

    ## Select only relevant inputs
    inputs = [int(i) - 1 for i in input_frames]
    images = [images[i] for i in inputs]

    if args.onnx:
        imgx = [net.get_inputs()[i].name for i in range(4)]
        intx = [net.get_outputs()[i].name for i in range(n_output)]
    else:
        imgx = ["img%d" % i for i in range(4)]

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            if args.onnx:
                output = net.run(intx,
                                 {k: v for k, v in zip(imgx, images)})
            else:
                output = net.predict({k: v for k, v in zip(imgx, images)})
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
            if i != 0:
                total_time = total_time + (end - start)
        logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
    else:
        if args.onnx:
            output = net.run(intx,
                             {k: v for k, v in zip(imgx, images)})
        else:
            output = net.predict({k: v for k, v in zip(imgx, images)})

    images = [postprocess(x[0]) for x in output]

    savepath = os.path.join(args.savepath, SAVE_IMAGE_PATH)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, images[0])

    logger.info('Script finished successfully.')


def main():
    info = {
        2: (WEIGHT_2x_PATH, MODEL_2x_PATH, 1),
        4: (WEIGHT_4x_PATH, MODEL_4x_PATH, 3),
        8: (WEIGHT_8x_PATH, MODEL_8x_PATH, 7),
    }
    weight_path, model_path, n_output = info[args.interpolation]
    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)
    else:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)

    recognize_from_image(net, n_output)


if __name__ == '__main__':
    main()
