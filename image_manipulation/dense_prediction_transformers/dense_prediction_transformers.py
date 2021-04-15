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
import torch
import numpy as np
from torchvision.transforms import Compose
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
def monodepth(optimize=True):
    # get input
    img_names = glob.glob(os.path.join('inputs', '*'))
    num_images = len(img_names)

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_MONODEPTH_PATH)
    else:
        net = ailia.Net(MODEL_MONODEPTH_PATH, WEIGHT_MONODEPTH_PATH, env_id=args.env_id)

    # create output folder
    os.makedirs('monodepth_outputs', exist_ok=True)

    for i, img_name in enumerate(img_names):
        """
            TODO: 縦横の可変長対応
            onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException:
            [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION :
            Non-zero status code returned while running Add node. Name:'Add_1391'
            Status Message: /Users/runner/work/1/s/onnxruntime/core/providers/cpu/math/element_wise_ops.h:487
            void onnxruntime::BroadcastIterator::Append(int64_t, int64_t) axis == 1 || axis == largest was false.
            Attempting to broadcast an axis by a dimension other than 1. 829 by 1105
        """
        img_raw = util.io.read_image(img_name)
        img = transform({"image": img_raw})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img).to(device).unsqueeze(0)
            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            sample = sample.to('cpu').detach().numpy().copy()

            if args.onnx:
                input_name = net.get_inputs()[0].name
                prediction = net.run(None, {input_name: sample.astype(np.float32)})
                prediction = prediction[0]
            else:
                prediction = net.predict(sample)

            prediction = torch.from_numpy(prediction).to(device)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_raw.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        filename = os.path.join(
            'monodepth_outputs', os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(filename, prediction, bits=2)

    return


def segmentation(optimize=True):
    # get input
    img_names = glob.glob(os.path.join('inputs', '*'))
    num_images = len(img_names)

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 384

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_SEGMENTATION_PATH)
    else:
        net = ailia.Net(MODEL_SEGMENTATION_PATH, WEIGHT_SEGMENTATION_PATH, env_id=args.env_id)

    # create output folder
    os.makedirs('segmentation_outputs', exist_ok=True)

    for i, img_name in enumerate(img_names):
        """
            TODO: 縦横の可変長対応
            onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException:
            [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION :
            Non-zero status code returned while running Add node. Name:'Add_1391'
            Status Message: /Users/runner/work/1/s/onnxruntime/core/providers/cpu/math/element_wise_ops.h:487
            void onnxruntime::BroadcastIterator::Append(int64_t, int64_t) axis == 1 || axis == largest was false.
            Attempting to broadcast an axis by a dimension other than 1. 829 by 1105
        """
        img_raw = util.io.read_image(img_name)
        img = transform({"image": img_raw})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img).to(device).unsqueeze(0)
            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            sample = sample.to('cpu').detach().numpy().copy()

            if args.onnx:
                input_name = net.get_inputs()[0].name
                prediction = net.run(None, {input_name: sample.astype(np.float32)})
                prediction = prediction[0]
            else:
                prediction = net.predict(sample)

            prediction = torch.from_numpy(prediction).to(device)
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=img_raw.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            prediction = torch.argmax(prediction, dim=1) + 1
            prediction = prediction.squeeze().cpu().numpy()


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
