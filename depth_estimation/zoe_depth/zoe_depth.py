import sys

import ailia
import cv2
import numpy as np
from einops import rearrange
from zoe_depth_util import get_params, save

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from arg_utils import (get_base_parser, get_savepath,  # noqa: E402
                       update_parser)
from model_utils import check_and_download_models

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/zoe_depth/'

IMAGE_PATH = './input.jpg'
SAVE_IMAGE_PATH = "./output.png"

MODEL_ARCHS = ("ZoeD_M12_K", "ZoeD_M12_N", "ZoeD_M12_NK")
INPUT_SIZE = {
    "ZoeD_M12_K": (768, 384),
    "ZoeD_M12_N": (512, 384),
    "ZoeD_M12_NK": (512, 384),
}

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('ZoeDepth model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch', default='ZoeD_M12_K', choices=MODEL_ARCHS,
    help='arch model.'
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
def infer_from_image(net):
    arch = args.arch
    input_image_file = args.input[0]
    img = cv2.imread(input_image_file)
    H, W = img.shape[:2]
    resized_img = cv2.resize(img, INPUT_SIZE[arch]).astype(np.float32) 
    resized_img /= 255.0
    resized_img_reversed = resized_img[..., ::-1]
    resized_img = rearrange(resized_img, "h w c -> 1 c h w")
    resized_img_reversed = rearrange(resized_img_reversed, "h w c -> 1 c h w")


    if args.onnx:
        input_name = net.get_inputs()[0].name
        output_name = net.get_outputs()[0].name
        pred_not_reversed = net.run([output_name], {input_name: resized_img})[0]
        pred_reversed = net.run([output_name], {input_name: resized_img_reversed})[0]
        pred = 0.5 * (pred_not_reversed + pred_reversed)
        pred = pred.squeeze()

    save(pred=pred, output_filename=SAVE_IMAGE_PATH, original_width=W, original_height=H)


def main():
    # model files check and download
    weight_path, model_path = get_params(args.arch)
    # check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # create net instance
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)
    else:
        net = ailia.Net(model_path, weight_path, env_id=args.env_id)

    # check input
    infer_from_image(net)


if __name__ == '__main__':
    main()
