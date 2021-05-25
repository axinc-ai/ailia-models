import sys, os
import time
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# import for dpt
import glob
import numpy as np
import util.io
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_MONODEPTH_PATH = "dpt_hybrid_monodepth.onnx"
MODEL_MONODEPTH_PATH = "dpt_hybrid_monodepth.onnx.prototxt"
WEIGHT_SEGMENTATION_PATH = "dpt_hybrid_segmentation.onnx"
MODEL_SEGMENTATION_PATH = "dpt_hybrid_segmentation.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dense_prediction_transformers/'


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Dense Prediction Transformers',
    None,
    None,
)
parser.add_argument(
    '--task',
    required=True,
    choices=['monodepth', 'segmentation'],
    help=('specify task you want to run.')
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img_raw):
    net_w = 576
    net_h = 384
    func = Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC)
    img = func({"image": img_raw})
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = normalization(img)
    img = PrepareForNet()(img)
    img = img["image"]
    return img


def monodepth(optimize=True):
    img_names = args.input

    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_MONODEPTH_PATH)
    else:
        net = ailia.Net(MODEL_MONODEPTH_PATH, WEIGHT_MONODEPTH_PATH, env_id=args.env_id)

    # create output folder
    os.makedirs('monodepth_outputs', exist_ok=True)

    for i, img_name in enumerate(img_names):
        img_raw = util.io.read_image(img_name)
        img = preprocess(img_raw)

        # compute
        sample = np.expand_dims(img,0)
        
        if args.onnx:
            input_name = net.get_inputs()[0].name
            prediction = net.run(None, {input_name: sample.astype(np.float32)})
            prediction = prediction[0]
        else:
            prediction = net.predict(sample)

        prediction = prediction[0]
        prediction = cv2.resize(prediction,(img_raw.shape[1],img_raw.shape[0]),interpolation=cv2.INTER_CUBIC)

        filename = os.path.join(
            'monodepth_outputs', os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(filename, prediction, bits=2)

    return


def segmentation(optimize=True):
    img_names = args.input

    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_SEGMENTATION_PATH)
    else:
        net = ailia.Net(MODEL_SEGMENTATION_PATH, WEIGHT_SEGMENTATION_PATH, env_id=args.env_id)

    # create output folder
    os.makedirs('segmentation_outputs', exist_ok=True)

    for i, img_name in enumerate(img_names):
        img_raw = util.io.read_image(img_name)
        img = preprocess(img_raw)

        # compute
        sample = np.expand_dims(img,0)

        if args.onnx:
            input_name = net.get_inputs()[0].name
            prediction = net.run(None, {input_name: sample.astype(np.float32)})
            prediction = prediction[0]
        else:
            prediction = net.predict(sample)
        prediction = prediction[0]

        scaled_predictin = np.zeros((prediction.shape[0],img_raw.shape[0],img_raw.shape[1]))
        for i in range(prediction.shape[0]):
            scaled_predictin[i] = cv2.resize(prediction[i],(img_raw.shape[1],img_raw.shape[0]),interpolation=cv2.INTER_CUBIC)
        prediction = np.argmax(scaled_predictin, axis=0) + 1

        filename = os.path.join(
            'segmentation_outputs', os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_segm_img(filename, img_raw, prediction, alpha=0.5)

    return


def main():
    if args.task == 'monodepth':
        check_and_download_models(WEIGHT_MONODEPTH_PATH, MODEL_MONODEPTH_PATH, REMOTE_PATH)
        monodepth()
    elif args.task == 'segmentation':
        check_and_download_models(WEIGHT_SEGMENTATION_PATH, MODEL_SEGMENTATION_PATH, REMOTE_PATH)
        segmentation()


if __name__ == '__main__':
    main()
